"""Figures for the README. Each function saves one PNG to results/figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from prefabrisk.risk import var_q

COLORS = {"historical": "#333333", "calibrated": "#1f77b4", "course": "#d62728",
          "summer": "#2ca02c", "autumn": "#ff7f0e"}


def plot_alpha_distribution(alphas, crew_index, crew_name, num_days, path):
    """Share of scenarios by completion coefficient of one crew.

    alphas[season] = {"historical": array, "calibrated": array, "course": array}
    """
    levels = np.arange(num_days + 1) / num_days
    fig, axes = plt.subplots(1, len(alphas), figsize=(11, 4), sharey=True)
    for ax, (season, sources) in zip(np.atleast_1d(axes), alphas.items()):
        for source, alpha in sources.items():
            days = np.rint(alpha[:, crew_index] * num_days).astype(int)
            share = np.bincount(days, minlength=num_days + 1) / len(days)
            style = dict(marker="o", markersize=3, linewidth=1.5, color=COLORS[source])
            if source == "historical":
                style.update(linewidth=2.5)
            ax.plot(levels, share, label=source, **style)
        ax.set_title(f"{season}")
        ax.set_xlabel(f"completion coefficient alpha of crew {crew_name}")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("share of 20-day cycles")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(f"Crew {crew_name}: course simulator vs Bielefeld 100 m wind", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_allocations(allocations, crew_names, path, title):
    """Grouped bars of first-stage allocation per crew, one bar per case."""
    cases = list(allocations)
    width = 0.8 / len(cases)
    positions = np.arange(len(crew_names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, case in enumerate(cases):
        ax.bar(positions + (k - (len(cases) - 1) / 2) * width, allocations[case], width,
               label=case, color=COLORS.get(case, None), edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"crew {n}" for n in crew_names])
    ax.set_ylabel("units contracted in advance")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_distributions(costs, q, threshold, path, title):
    """Histograms of realised total cost, simulated vs historical, with VaR.

    costs[season] = {"simulated": array, "historical": array}
    """
    fig, axes = plt.subplots(1, len(costs), figsize=(11, 4), sharey=True)
    all_costs = np.concatenate([c for sources in costs.values() for c in sources.values()])
    bins = np.linspace(all_costs.min(), all_costs.max(), 40)
    for ax, (season, sources) in zip(np.atleast_1d(axes), costs.items()):
        ax.hist(sources["simulated"], bins=bins, density=True, alpha=0.5,
                color=COLORS[season], label="simulated (10,000)")
        ax.hist(sources["historical"], bins=bins, density=True, histtype="step", linewidth=2,
                color=COLORS["historical"], label=f"historical ({len(sources['historical'])})")
        for source, style in [("simulated", "--"), ("historical", ":")]:
            ax.axvline(var_q(sources[source], q), color="black", linestyle=style, linewidth=1.2,
                       label=f"{q}% VaR {source}")
        ax.axvline(threshold, color="grey", linewidth=1, label=f"EUR {threshold:,}")
        ax.set_title(season)
        ax.set_xlabel("total realised cost (thousand EUR)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)