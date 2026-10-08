import jax.numpy as jnp

from workshop.m07_gradients import gradient_variance_study, laplace
from tests._util import key
from tests.m07._common import log_joint


def test_step5_variance_scales_like_one_over_n():
    mean, cov = laplace(log_joint, jnp.zeros(5))
    params = {"loc": mean + 0.3, "log_scale": jnp.log(jnp.sqrt(jnp.diag(cov))) + 0.2}
    ns = (4, 16, 64, 256)
    study = gradient_variance_study(key(330), params, log_joint, ns, n_repeats=150)
    assert set(study) == {"score", "score_cv", "reparam"}
    for name, v in study.items():
        assert v.shape == (4,)
        assert bool(jnp.all(jnp.diff(v) < 0)), name
        ratio = float(v[0] / v[-1])
        assert 20 < ratio < 200, (name, ratio)  # ideal 64
    assert bool(jnp.all(study["reparam"] < study["score_cv"]))
    assert bool(jnp.all(study["score_cv"] < study["score"]))
