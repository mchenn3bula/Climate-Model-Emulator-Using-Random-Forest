"""Headless diagnostic plots; synthetic output is labeled on the figures themselves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(predictions, summary, output, evidence, seed):
    title = (
        "SYNTHETIC DEMO — not climate performance"
        if evidence == "synthetic"
        else "Climate emulation holdout evaluation"
    )
    families = ["mean", "ridge", "forest", "scaled_forest"]
    regimes = list(dict.fromkeys(row["regime"] for row in summary))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x, width = np.arange(len(regimes)), 0.18
    for i, family in enumerate(families):
        rows = [
            next(r for r in summary if r["family"] == family and r["regime"] == regime)
            for regime in regimes
        ]
        ax.bar(
            x + (i - 1.5) * width,
            [r["rmse_mean"] for r in rows],
            width,
            yerr=[r["rmse_std"] or 0 for r in rows],
            label=family,
            capsize=2,
        )
    ax.set(
        xticks=x,
        xticklabels=[r.replace("_", " ") for r in regimes],
        ylabel="RMSE (target units)",
        title=title,
    )
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "rmse_comparison.png", dpi=160)
    plt.close(fig)
    part = predictions[predictions.model == f"forest_seed{seed}"]
    # Aggregate multiple scenarios at each location for a readable location diagnostic.
    part = part.assign(residual=part.prediction - part.tas_FINAL)
    grouped = part.groupby(["regime", "lat", "lon"], as_index=False).residual.mean()
    limit = max(float(grouped.residual.abs().max()), 1e-8)
    fig, axes = plt.subplots(1, len(regimes), figsize=(13, 4), layout="constrained", squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        points = grouped[grouped.regime == regime]
        image = ax.scatter(
            points.lon, points.lat, c=points.residual, cmap="RdBu_r", vmin=-limit, vmax=limit, s=22
        )
        ax.set(title=regime.replace("_", " "), xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(image, ax=list(axes[0]), label="Mean prediction − target")
    fig.suptitle(f"{title}\nRandom Forest residuals, seed {seed}")
    fig.savefig(output / "residual_locations.png", dpi=160)
    plt.close(fig)
