"""Change-of-variables checker, provided (not a stub).

Every time this workshop moves a density from one coordinate system to another, a
log-Jacobian term appears. Forgetting it is silent: nothing crashes, the posterior is
simply wrong. The two functions here let you test any such transform in three lines.
They are used on the pages of Modules 2b, 5, 8, 9, 11, 13 and 15a.

Usage (constrained lambda > 0, unconstrained phi = log lambda):

    from workshop.jacobian_check import check_change_of_variables, integral_1d
    log_p_lam = lambda lam: gamma_log_pdf(lam, 2.0, 0.5)
    log_p_phi = lambda phi: log_p_lam(jnp.exp(phi)) + phi        # claimed density on phi
    check_change_of_variables(log_p_lam, jnp.exp, log_p_phi, jnp.linspace(-2, 3, 11))
    # -> array of zeros (to 1e-8) if log_p_phi is right; the missing Jacobian otherwise
    integral_1d(log_p_phi, jnp.linspace(-8, 6, 20001))           # -> 1.0
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def log_abs_det_jacobian(forward: Callable[[Array], Array], z: Array) -> Array:
    """log |det d forward / d z| at a single point z (scalar or 1-D vector)."""
    z = jnp.atleast_1d(jnp.asarray(z))
    jac = jax.jacfwd(lambda v: jnp.atleast_1d(forward(v.reshape(z.shape) if z.ndim else v[0])))(z)
    jac = jac.reshape(z.size, z.size)
    return jnp.linalg.slogdet(jac)[1]


def check_change_of_variables(
    log_density_x: Callable[[Array], Array],
    forward: Callable[[Array], Array],
    log_density_z: Callable[[Array], Array],
    z_points: Array,
) -> Array:
    """Discrepancy of a claimed density in transformed coordinates.

    `forward` maps the new coordinate z to the original coordinate x = forward(z).
    `log_density_x` is the (known) log density of x; `log_density_z` is your claimed log
    density of z. The correct relation is

        log p_z(z) = log p_x(forward(z)) + log |det J_forward(z)|.

    Returns, for each row of `z_points`, the difference between the left and right
    sides. A correct implementation returns zeros to numerical precision. A missing
    Jacobian returns exactly -log|det J| at each point, which for z = log x is -z.
    """
    z_points = jnp.asarray(z_points)
    if z_points.ndim == 1:
        z_points = z_points[:, None]

    def one(z):
        zz = z[0] if z.shape == (1,) else z
        rhs = log_density_x(forward(zz)) + log_abs_det_jacobian(forward, z)
        return jnp.squeeze(log_density_z(zz)) - jnp.squeeze(rhs)

    return jax.vmap(one)(z_points)


def integral_1d(log_density: Callable[[Array], Array], grid: Array) -> Array:
    """Trapezoidal integral of exp(log_density) over a fine 1-D grid.

    For a correctly normalised density this is 1. A density that forgot a Jacobian is
    off by the average of the Jacobian under the right density, which is almost never 1.
    """
    grid = jnp.asarray(grid)
    vals = jnp.exp(jax.vmap(log_density)(grid))
    return jnp.trapezoid(vals, grid)
