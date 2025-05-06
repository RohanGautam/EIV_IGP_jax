import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .utils import evaluate_kernel, integrate_kernel, invert


def eiv_igp(x, x_std, y, y_std, x_star, simplify=False):
    alpha = numpyro.sample("alpha", dist.Normal(0, 1000.0))
    p = numpyro.sample("p", dist.Uniform(0.0, 1.0))
    kernel_precision = numpyro.sample("tau_g", dist.Gamma(10.0, 100.0))
    kernel_variance = 1 / kernel_precision
    microscale_std = numpyro.sample("sigma", dist.Uniform(0.01, 1.0))
    params = {"p": p, "kappa": 1.99, "variance": kernel_variance}

    C_w = evaluate_kernel(x_star[:, None], x_star[None, :], params)
    C_w += jnp.eye(len(x_star)) * 1e-5  # jitter

    # sample rates
    w_m = numpyro.sample("w_m", dist.MultivariateNormal(jnp.zeros(len(x_star)), C_w))

    Cw_inv_w_m = jnp.matmul(invert(C_w), w_m)

    if simplify:
        # quadrature over fixed x
        K_hw = integrate_kernel(x, x_star, params)
    else:
        # quadrature over estimated x (EIV)
        # x_true = numpyro.sample("x_true", dist.Normal(x, x_std).to_event(1))
        x_true = numpyro.sample("x_true", dist.Normal(0, 1e3**0.5).expand([len(x)]))
        K_hw = integrate_kernel(x_true, x_star, params)
        numpyro.sample(
            "x",
            dist.Normal(loc=x_true, scale=x_std).to_event(1),
            obs=x,
        )

    h_x = jnp.matmul(K_hw, Cw_inv_w_m)

    # track mean
    integrated_mean = numpyro.deterministic("integrated_mean", alpha + h_x)

    numpyro.sample(
        "y",
        dist.Normal(
            integrated_mean,
            jnp.sqrt(y_std**2 + microscale_std**2),
        ).to_event(1),
        obs=y,
    )


def get_predictions_on_grid(samples, grid):
    derivative_process_samples = samples["w_m"]
    posterior_predictive = numpyro.infer.Predictive(eiv_igp, samples)(
        jax.random.PRNGKey(1),
        x=grid,
        x_star=grid,
        # these are not used for calculating integrated mean predictive
        x_std=jnp.zeros_like(grid),
        y=jnp.zeros_like(grid),
        y_std=jnp.zeros_like(grid),
        simplify=True,  # is True regardless, to ignore x_true recomputation
    )
    integrated_process_mean_samples = posterior_predictive["integrated_mean"]
    return derivative_process_samples, integrated_process_mean_samples
