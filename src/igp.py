import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .utils import evaluate_kernel, integrate_kernel, invert


def eiv_igp(x, y, cov, x_star, simplify=False, **kwargs):
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

    x_true = numpyro.sample("x_true", dist.Normal(0, 1e3**0.5).expand([len(x)]))

    if simplify:
        # quadrature over fixed x
        K_hw = integrate_kernel(x, x_star, params)
    else:
        # quadrature over estimated x (EIV)
        K_hw = integrate_kernel(x_true, x_star, params)

    h_x = jnp.matmul(K_hw, Cw_inv_w_m)

    # track mean
    integrated_mean = numpyro.deterministic("integrated_mean", alpha + h_x)

    # add microscale variance to the cov matrix
    cov_updated = cov + jnp.array([[0, 0], [0, microscale_std**2]])
    mean = jnp.c_[x_true, integrated_mean]
    obs = jnp.c_[x, y]

    with numpyro.plate("obs_plate", len(obs)):
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(loc=mean, covariance_matrix=cov_updated),
            obs=obs,
        )


def get_predictions_on_grid(samples, eiv_input):
    samples = samples.copy()
    grid = eiv_input["x_star"]
    # just fix the shape of x_true samples - does not effect variables of interest we want to extract here
    samples["x_true"] = jnp.zeros((len(samples["x_true"]), len(grid)))

    derivative_process_samples = samples["w_m"]
    posterior_predictive = numpyro.infer.Predictive(eiv_igp, samples)(
        jax.random.PRNGKey(1),
        x=grid,
        x_star=grid,
        # these are not used for calculating integrated mean predictive
        # x_std=jnp.zeros_like(grid),
        cov=jnp.zeros((len(grid), 2, 2)),
        y=jnp.zeros_like(grid),
        simplify=True,  # is True here to doubly integrate over x_star
    )
    integrated_process_mean_samples = posterior_predictive["integrated_mean"]
    return derivative_process_samples, integrated_process_mean_samples
