import jax
import jax.numpy as jnp


@jax.jit
def evaluate_kernel(x, y, params):
    """
    Evaluate the kernel. Can be for a single point,
    or over a grid, based on how broadcasting is handled for arguments passed.
    """
    p, kappa, sigma = params["p"], params["kappa"], params["variance"]
    # absolute distance is same as euclidean distance in 1D
    diff = jnp.abs(x - y)
    # powered exponential
    return sigma * (p ** (diff**kappa))


@jax.jit
def invert(matrix):
    """
    Invert a positive-definite matrix (a covariance matrix fits this property) using stable Cholesky decomposition.
    """
    L = jax.scipy.linalg.cho_factor(matrix, lower=True)
    identity = jnp.eye(matrix.shape[0])
    matrix_inv = jax.scipy.linalg.cho_solve(L, identity)
    return matrix_inv


@jax.jit
def integrate_kernel(x, y, kernel_params):
    """
    integrate kernel over upper limit values given by x. Uses Chebyshev-Gauss quadrature
    """

    integration_len = 30
    # [-1,1]
    chebyshev_nodes = jnp.cos(
        jnp.pi * (0.5 + jnp.arange(0, integration_len)) / integration_len
    )

    # set up upper and lower bounds. Lower one is just for clarity.
    # set last dim to be 1
    a, b = jnp.zeros_like(x)[:, None], x[:, None]

    # scale chebyshev nodes to [a,b]
    # shape (len(x), integration_len)
    f_input = ((b - a) / 2) * chebyshev_nodes[None, :] + ((b - a) / 2) + a

    # shape (len(x), len(y), integration_len)
    f_eval = evaluate_kernel(f_input[:, None, :], y[None, :, None], kernel_params)
    weights = jnp.pi / integration_len
    jacobian = (b - a) / 2
    chebyshev_func_form_coeff = jnp.sqrt(1 - chebyshev_nodes**2)

    # heavy use of broadcasting, but is very convenient and cool once you get it.
    final_term = (
        weights
        * f_eval
        * jacobian[:, None, :]
        * chebyshev_func_form_coeff[None, None, :]
    )
    # (len(x),len(y))
    return jnp.sum(final_term, axis=-1)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = {"p": 0.5, "kappa": 1.99, "variance": 1}
    x_grid = jnp.linspace(-3.0, 3.0, 100)
    K = evaluate_kernel(x_grid[:, None], x_grid[None, :], params)

    # Plot the kernel matrix
    plt.imshow(K, extent=(-3, 3, -3, 3), origin="lower", cmap="viridis")
    plt.show()

    matrix = jnp.array([[4.0, 2.0], [2.0, 3.0]])
    matrix_inv = invert(matrix)
    print(matrix_inv @ matrix)  # ~Identity

    out = integrate_kernel(jnp.zeros(35), jnp.zeros(10), params)
    print(out.shape)
