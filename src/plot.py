import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_input(df: pd.DataFrame):
    """
    Plots sea-level data with error boxes, applying a GIA correction if specified.

    Args:
        df (pd.DataFrame): Input DataFrame.
        gia_rate (float): GIA correction rate in m/kilo-year.
        title (str): Title for the plot.
        y_label (str): Label for the y-axis.
        fig_label (str): Label for the figure (e.g., 'B', 'C').
    """

    x = df["Age"]
    y = df["RSL"]
    x_std = df["AgeError"]
    y_std = df["RSLError"]

    x_err_2sigma = 2 * x_std
    y_err_2sigma = 2 * y_std

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot individual data points as open black circles
    plt.plot(
        x,
        y,
        "o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=6,
        zorder=2,
    )

    # Plot error boxes (2-sigma uncertainty)
    for i in range(len(x)):
        # Calculate the bottom-left corner, width, and height for the rectangle
        rect_x_min = x[i] - x_err_2sigma[i] / 2
        rect_y_min = y[i] - y_err_2sigma[i] / 2
        rect_width = x_err_2sigma[i]
        rect_height = y_err_2sigma[i]

        rect = patches.Rectangle(
            (rect_x_min, rect_y_min),
            rect_width,
            rect_height,
            linewidth=0.5,
            edgecolor="gray",
            facecolor="none",
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(rect)


if __name__ == "__main__":
    df = pd.read_csv("../data/NYC.csv")
    plot_input(df)
    plt.show()
