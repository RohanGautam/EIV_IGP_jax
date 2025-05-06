import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import numpyro
import numpyro.distributions as dist
import arviz as az

print(jax.devices())
jax.config.update("jax_enable_x64", True)


def evaluate_kernel(x, y, kernel_fn, params):
    K = jax.vmap(lambda x1: jax.vmap(lambda y1: kernel_fn(x1, y1, params))(y))(x)
    return K


def powered_exponential(x, y, params):
    p, kappa, sigma = params["p"], params["kappa"], params["variance"]
    return sigma * (p ** (jnp.abs(x - y) ** kappa))


def invert(matrix):
    """
    Inverts a positive-definite matrix using Cholesky decomposition.
    """
    # Perform Cholesky decomposition
    L = jax.scipy.linalg.cho_factor(matrix, lower=True)

    # Use Cholesky factors to compute the inverse
    identity = jnp.eye(matrix.shape[0])
    matrix_inv = jax.scipy.linalg.cho_solve(L, identity)

    return matrix_inv


def igp(x_star, quad1, quad2, y, y_std):
    # N = x.shape[0]
    alpha = numpyro.sample("beta0", dist.Normal(0, 1000.0))
    p = numpyro.sample("p", dist.Uniform(0.0, 1.0))
    # p = numpyro.deterministic("p", 0.005)
    kernel_precision = numpyro.sample("tau_g", dist.Gamma(10.0, 100.0))
    kernel_variance = 1 / kernel_precision
    microscale_std = numpyro.sample("sigma", dist.Uniform(0.01, 1.0))
    params = {"p": p, "kappa": 1.99, "variance": 1}

    # kernel computation and EIV sampling
    # x_true = numpyro.sample("x_true", dist.Normal(x, x_std).to_event(1))

    C_w = evaluate_kernel(x_star, x_star, powered_exponential, params)
    C_w += jnp.eye(len(x_star)) * 1e-5  # jitter
    # sample rates
    # Sample w_m (rates on grid) from its prior MVN(0, K_ww)
    w_m = numpyro.sample(
        "w_m", dist.MultivariateNormal(jnp.zeros(len(x_star)), C_w * kernel_variance)
    )

    # Calculate C_w(-1)w_m
    # Cw_inv_w_m = jax.scipy.linalg.cho_solve(
    #     jax.scipy.linalg.cho_factor(C_w, lower=True),
    #     w_m,
    # )
    Cw_inv_w_m = jnp.matmul(invert(C_w), w_m)

    # integral approximation
    # C_w is multiplied by variance, but in its inverse has the reciprocal of variance
    # set var as 1 and avoid variance mult
    K_hw = jnp.sum(p**quad1 * quad2, axis=2)  # * kernel_variance
    h_x = jnp.matmul(K_hw, Cw_inv_w_m)

    # h_integral_cond_exp = jnp.matmul(K_hw, K_ww_inv_w_m)
    with numpyro.plate("y_plate", len(y)):
        # numpyro.sample(
        #     "x_likelihood",
        #     dist.Normal(loc=x_true, scale=x_std),
        #     obs=x,
        # )  # not used, but used in eivigp_estquad.txt.
        numpyro.sample(
            "y",
            dist.Normal(alpha + h_x, jnp.sqrt(y_std**2 + microscale_std**2)),
            obs=y,
        )
