import jax
import jax.numpy as jnp
import numpy as np

from workshop.m09_hmc import uplift_noncentered_log_prob
from workshop.m12_handlers import uplift_model
from workshop.m13_ppl_inference import unconstrain, unconstrained_log_density
from tests.m13._common import SIGMA, Y
from tests._util import key


def test_step2_equals_module9_target_up_to_constant_with_exact_gradients():
    spec = unconstrain(uplift_model, Y, SIGMA)
    f = lambda z: unconstrained_log_density(uplift_model, spec, z, Y, SIGMA)
    z1, z2 = jax.random.normal(key(41), (10,)), jax.random.normal(key(42), (10,))
    d_ppl = f(z1) - f(z2)
    d_ref = uplift_noncentered_log_prob(z1) - uplift_noncentered_log_prob(z2)
    np.testing.assert_allclose(d_ppl, d_ref, rtol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(f))(z1), jax.grad(uplift_noncentered_log_prob)(z1), rtol=1e-10, atol=1e-12)


def test_step2_jacobian_term_is_present():
    # Without the Jacobian the derivative w.r.t. log tau would miss the +1 from d(log tau)/d(log tau).
    from workshop.m12_handlers import log_density
    from workshop.m13_ppl_inference import unflatten

    spec = unconstrain(uplift_model, Y, SIGMA)
    z = jax.random.normal(key(43), (10,))
    with_j = unconstrained_log_density(uplift_model, spec, z, Y, SIGMA)
    without_j = log_density(uplift_model, unflatten(z, spec), Y, SIGMA)[0]
    np.testing.assert_allclose(with_j - without_j, z[1], rtol=1e-12)


def test_step2_multidimensional_site_with_vector_bijector():
    # A (2, 3) site whose support bijector has a length-3 scale: the log-Jacobian must be
    # evaluated on the site's shape, not on the flat slice.
    from workshop.m12_handlers import Uniform, log_density, sample
    from workshop.m13_ppl_inference import unflatten

    high = jnp.array([1.0, 2.0, 4.0])

    def model():
        sample("w", Uniform(jnp.zeros(3), high), sample_shape=(2,))

    spec = unconstrain(model)
    z = jax.random.normal(key(44), (6,))
    got = unconstrained_log_density(model, spec, z)
    zz = z.reshape(2, 3)
    expected = log_density(model, unflatten(z, spec))[0] + jnp.sum(
        jax.nn.log_sigmoid(zz) + jax.nn.log_sigmoid(-zz) + jnp.log(high)
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)
