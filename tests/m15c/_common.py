import jax.numpy as jnp
import numpy as np

from workshop.data import AdSpendSCM, ad_spend_observational

SCM = AdSpendSCM()
DATA = {k: jnp.asarray(v) for k, v in ad_spend_observational().items()}


def structural_matrix_reference(scm: AdSpendSCM) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy construction of x = M u for x = (S, A, T, C), u = (S, U_A, U_T, U_C)."""
    M = np.zeros((4, 4))
    M[0, 0] = 1.0
    M[1] = scm.a_s * M[0] + np.array([0, 1, 0, 0])
    M[2] = scm.b_s * M[0] + scm.b_a * M[1] + np.array([0, 0, 1, 0])
    M[3] = scm.c_t * M[2] + scm.c_s * M[0] + np.array([0, 0, 0, 1])
    return M, np.array([1.0, scm.sd_a, scm.sd_t, scm.sd_c])


def row(i: int) -> dict[str, jnp.ndarray]:
    return {k: DATA[k][i] for k in ("S", "A", "T", "C", "U_A", "U_T", "U_C")}
