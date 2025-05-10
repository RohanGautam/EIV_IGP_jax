import pandas as pd
import numpy as np


df = pd.read_csv("./data/NYC.csv")


def preprocess(df, gia_rate=0, y_occ=2010):
    x = df["Age"].to_numpy() / 1000
    x_std = df["AgeError"].to_numpy() / 1000
    y = df["RSL"].to_numpy()
    y_std = df["RSLError"].to_numpy()
    y_occ = y_occ / 1000
    interval = 30 / 1000

    # if gia_rate is not None:
    #     y_occ /= 1000
    #     y += (y_occ - x) * gia_rate

    x_grid = np.concatenate(
        [
            [(x - x_std).min()],
            np.arange(x.min(), x.max(), interval),
            [(x + x_std).max()],
        ]
    )
    x_star = x_grid - x.min()
    x = x - x.min()

    # gia correction (no effect if rate is 0)
    A = np.array([[1, 0], [-gia_rate, 1]])
    y += (y_occ - x) * gia_rate  # change from Azi+b
    cov = np.array([[[x**2, 0], [0, y**2]] for x, y in zip(x_std, y_std)])
    cov = A @ cov @ A.T

    return {
        "x": x,
        # "x_std": x_std,
        "y": y,
        # "y_std": y_std,
        "cov": cov,
        "x_star": x_star,
    }
