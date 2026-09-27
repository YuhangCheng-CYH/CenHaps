"""JAX/Flax deconvolution model and training loop.

The model predicts observed k-mer counts as a linear combination of
haplotype expression profiles. At non-root tree levels, proportions are
decomposed hierarchically: learnable parent masses (warm-started from
the parent level's props) x intra-group softmax. The parent level's
solution acts as an initialization, not a hard constraint.

Signal (the scale factor bridging ref density and NGS count) is computed
in closed form as the MSE-optimal scaling:

    signal = <obs, raw_pred> / <raw_pred, raw_pred>

This is NOT a learnable parameter. It is recomputed from the data at every
forward pass, so the optimizer only adjusts props (the direction of the
haplotype combination). This eliminates the signal-props identifiability
degeneracy: when signal was free, a regularizer could push props toward
one-hot while signal compensated, producing correct reconstruction but
wrong proportions. With closed-form signal, the optimizer cannot "cheat"
by adjusting signal to mask bad props.

A top-k sparsity penalty acts as a tie-breaker: it penalizes only the
probability mass BEYOND the top-k groups (k=2 for diploid), preserving
the model's ability to express heterozygous (2-haplotype) samples while
suppressing noise groups. Pure homozygous ([1,0,...]) and heterozygous
([0.5,0.5,0,...]) both incur zero penalty; only 3+ peak distributions
are pushed back toward 2. The weight is constant (no annealing), since
the penalty direction is already correct.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax


class DeconvModel(nn.Module):
    """Predict raw_pred = dot(hap_exprs, props); signal computed in closed form.

    The model only learns props (haplotype proportions). Signal is not a
    parameter — it is derived from the data at each forward pass.
    """
    n_haps: int
    group_assign: tuple = ()
    previous_prop: tuple = ()

    @nn.compact
    def __call__(self, hap_exprs):
        if not self.group_assign:
            # Root level: single softmax over all haplotypes
            init_props = self.param(
                "init_props", nn.initializers.normal(0.1), (self.n_haps,)
            )
            props = jax.nn.softmax(init_props)
        else:
            # Child level: LEARNABLE parent masses (warm-started from
            # previous_prop) x intra-group softmax.
            # previous_prop acts as a soft prior / initialization, NOT a
            # hard constraint: parent masses can shift across groups when
            # the data disagrees with the parent level's decision.
            prior = jnp.asarray(self.previous_prop, dtype=jnp.float32)
            # Floor so groups pushed to ~0 by top-k sparsity can revive:
            # at p=1e-8 the softmax gradient p(1-p) ~ 1e-8 and the group
            # stays frozen. 1e-4 keeps a gradient channel open while
            # distorting the warm start by <0.5% (48 groups x 1e-4).
            prior = jnp.maximum(prior, 1e-4)
            prior = prior / jnp.sum(prior)
            # softmax(log(p)) == p, so step 0 reproduces the parent
            # solution exactly.
            init_parent_logits = jnp.log(prior)

            parent_logits = self.param(
                "parent_logits",
                lambda rng, shape: init_parent_logits,
                (len(self.previous_prop),),
            )
            parent_props = jax.nn.softmax(parent_logits)

            props = jnp.zeros(self.n_haps)
            for group in range(len(self.previous_prop)):
                members = [
                    m for m in range(self.n_haps)
                    if self.group_assign[m] == group
                ]
                init_props = self.param(
                    f"init_props_{group}",
                    nn.initializers.normal(0.1),
                    (len(members),),
                )
                intra = jax.nn.softmax(init_props)
                for idx, hap in enumerate(members):
                    props = props.at[hap].set(
                        parent_props[group] * intra[idx]
                    )

        # raw_pred = direction vector (which haplotypes, in what proportion)
        # signal is NOT computed here — it needs obs_count, which is only
        # available in the training loop. See closed_form_signal() below.
        raw_pred = jnp.dot(hap_exprs, props)
        return raw_pred, props


def closed_form_signal(obs_count, raw_pred):
    """MSE-optimal scale factor: signal = <obs, raw_pred> / <raw_pred, raw_pred>.

    Derivation: minimize ||signal * raw_pred - obs||^2 w.r.t. signal.
    Taking derivative and setting to 0 gives the closed-form solution.

    This bridges the scale gap between ref k-mer density (hap_expr) and
    NGS count (obs_count) without introducing a free parameter that could
    trade off with props. The effective MSE becomes:

        MSE = ||obs||^2 * sin^2(theta)

    where theta is the angle between raw_pred and obs. The model optimizes
    only the direction (props), not the magnitude (signal).
    """
    # Ensure 1D: polars single-column .to_numpy() returns (n, 1)
    obs_count = jnp.ravel(obs_count)
    raw_pred = jnp.ravel(raw_pred)
    dot_or = jnp.dot(obs_count, raw_pred)
    dot_rr = jnp.dot(raw_pred, raw_pred)
    return dot_or / (dot_rr + 1e-8)


def create_train_state(model, rng, hap_exprs, learning_rate=5e-3,
                       n_steps=4000):
    """Create Flax TrainState with Adam + cosine decay lr schedule.

    Adam moves each parameter ~lr per step (m/sqrt(v) ~ +/-1), so
    reaching a one-hot softmax (logit gap ~4.6) needs ~4.6/lr steps:
    ~4600 at lr=1e-3 (why 4000 steps never converged), ~900 at 5e-3.
    Cosine decay to 2% (alpha=0.02) travels fast early, settles finely
    late.
    """
    params = model.init(rng, hap_exprs)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=n_steps, alpha=0.02
    )
    tx = optax.adam(schedule)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


@jax.jit
def train_step(state, hap_exprs, obs_count, sparsity_weight, top_k=2):
    """MSE + top-k sparsity: penalize prob mass BEYOND the top-k groups.

    Preserves heterozygous (2-haplotype) expression while suppressing
    noise groups. When n_haps <= top_k, the penalty is 0.
    """
    def loss_fn(params):
        raw_pred, props = state.apply_fn(params, hap_exprs)
        signal = closed_form_signal(obs_count, raw_pred)
        predicted_count = raw_pred * signal
        mse = jnp.mean((predicted_count - obs_count) ** 2)

        # Quality Beyond Top-K: Sort in ascending order and compute the sum only for the smallest (n-k) elements
        sorted_asc = jnp.sort(props)
        n = props.shape[0]
        mask = jnp.arange(n) < jnp.maximum(n - top_k, 0)  # True=Not top-k
        rest_mass = jnp.sum(jnp.where(mask, sorted_asc, 0.0))
        return mse + sparsity_weight * rest_mass

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_value


def training_model(hap_expr, obs_count, n_haps, group_assign, previous_prop,
                   random_seed, n_steps, sparsity_weight=0.01, top_k=2,
                   previous_signal=1.0, learning_rate=5e-3):
    model = DeconvModel(n_haps=n_haps, group_assign=group_assign,
                        previous_prop=previous_prop)
    rng = jax.random.PRNGKey(random_seed)
    state = create_train_state(model, rng, hap_expr, learning_rate, n_steps)

    prev_props = None
    stable_checks = 0
    for step in range(n_steps):
        sw = jnp.float32(sparsity_weight)
        state, loss = train_step(state, hap_expr, obs_count, sw, top_k)
        if step % 200 == 0:
            raw_pred, props = state.apply_fn(state.params, hap_expr)
            signal = closed_form_signal(obs_count, raw_pred)
            print(f"Step {step:4d} | Loss: {loss:.6f} | "
                  f"Props: {props} | Signal: {signal:.4f}")
            # Early stop on PROPS stability (not loss): loss plateaus
            # long before props settle, and deep levels hit an
            # irreducible noise floor where extra steps are wasted.
            # Stop when max|delta props| < 1e-3 for 3 consecutive
            # checks (600 steps).
            if prev_props is not None:
                delta = float(jnp.max(jnp.abs(props - prev_props)))
                stable_checks = stable_checks + 1 if delta < 1e-3 else 0
                if stable_checks >= 3:
                    print(f"Early stop at step {step} "
                          f"(props stable, max delta {delta:.2e})")
                    break
            prev_props = props

    raw_pred, final_props = state.apply_fn(state.params, hap_expr)
    final_signal = closed_form_signal(obs_count, raw_pred)
    final_pre = raw_pred * final_signal
    final_loss = jnp.mean((final_pre - obs_count) ** 2)
    return final_pre, final_props, final_signal, final_loss
