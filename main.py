import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd

from src import data, eivigp
from src import plot as plot_utils

jax.config.update("jax_enable_x64", True)


def main(input_file):
    output_folder = Path("out")
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    eiv_input = data.preprocess(df, gia_rate=0)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            eivigp.eiv_igp,
            dense_mass=True,
        ),
        num_warmup=200,
        num_samples=800,
        num_chains=2,
    )
    mcmc.run(
        jax.random.PRNGKey(1),
        **eiv_input,
    )
    mcmc.print_summary()

    ## post processing
    samples = mcmc.get_samples()
    derp, intp = eivigp.get_predictions_on_grid(samples, eiv_input)

    ## plot
    print("Saving plots")
    plot_utils.plot_input(df, points=False)
    # plots mean+2sd
    plot_utils.plot_samples(eiv_input, intp)
    plt.title("Sea level curve")
    plt.xlabel("Time (AD)")
    plt.savefig(output_folder / "sea_level_curve.png")
    plt.figure(figsize=(10, 5))
    plt.clf()
    plot_utils.plot_samples(eiv_input, derp)
    plt.title("Sea level rate curve")
    plt.xlabel("Time (AD)")
    plt.savefig(output_folder / "sea_level_rate_curve.png")

    ## Save to csv
    print("Saving outputs")
    sl_data = {
        "x": eiv_input["x_unscaled"],
        "y": np.mean(intp, axis=0),
        "y_1sd": np.std(intp, axis=0),
    }
    rate_data = {
        "x": eiv_input["x_unscaled"],
        "y": np.mean(derp, axis=0),
        "y_1sd": np.std(derp, axis=0),
    }
    pd.DataFrame(sl_data).to_csv(output_folder / "sl.csv")
    pd.DataFrame(rate_data).to_csv(output_folder / "rate.csv")


if __name__ == "__main__":
    input_file = "./data/NYC.csv"
    main(input_file)
