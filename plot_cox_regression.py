# Creates an interpretable forest plot from the saved Cox model summary.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "cox_regression_summary.csv"
OUTPUT = ROOT / "cox_regression_forest_plot.png"
TIMING_OUTPUT = ROOT / "cox_regression_release_timing_plot.png"


def create_release_timing_plot(results: pd.DataFrame) -> None:
    """Plot release-year and release-month estimates separately."""
    results = results[
        results["term"].str.startswith("C(release_month)")
        | results["term"].eq("release_year_decade")
    ].copy()

    def make_label(term: str) -> str:
        if term == "release_year_decade":
            return "Release year (+10 years)"
        value = term.split("[T.", 1)[1].rstrip("]")
        month_names = {
            "2": "February", "3": "March", "4": "April", "5": "May",
            "6": "June", "7": "July", "8": "August", "9": "September",
            "10": "October", "11": "November", "12": "December",
        }
        return f"Release month: {month_names[value]}"

    results["label"] = results["term"].map(make_label)
    results["hr"] = results["coef"].map(__import__("math").exp)
    results["lo"] = results["coef lower 95%"].map(__import__("math").exp)
    results["hi"] = results["coef upper 95%"].map(__import__("math").exp)
    results["significant"] = results["p"] < 0.05
    # Order months chronologically, followed by the continuous release-year effect.
    results["group_order"] = results["term"].eq("release_year_decade").astype(int)
    results["month_order"] = results["term"].map(
        lambda term: int(term.split("[T.", 1)[1].rstrip("]"))
        if term.startswith("C(release_month)") else 0
    )
    results = results.sort_values(["group_order", "month_order"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    y = range(len(results))
    colors = ["#147D64" if sig else "#8A94A6" for sig in results["significant"]]
    for yi, (_, row), color in zip(y, results.iterrows(), colors):
        ax.errorbar(
            row["hr"], yi,
            xerr=[[row["hr"] - row["lo"]], [row["hi"] - row["hr"]]],
            fmt="o", color=color, ecolor=color, capsize=2, markersize=4,
            linewidth=1.2,
        )

    ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 10)
    ax.set_yticks(list(y), results["label"])
    ax.invert_yaxis()
    ax.axhline(10.5, color="#B8C0CC", linewidth=0.9)
    ax.set_xlabel("Adjusted hazard ratio (95% confidence interval; log scale)")
    ax.set_title("Release timing and Top 10 exit", loc="left", pad=14)
    ax.grid(axis="x", which="major", color="#D9DEE7", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.text(
        0.01, -0.14,
        "Months compare with January (the reference month, so it has no dot). "
        "Above 1 = faster exit; below 1 = slower exit. Green: p < .05.",
        transform=ax.transAxes, fontsize=8.5, color="#4B5563", va="top",
    )
    ax.text(
        0.01, -0.21,
        "Associations are adjusted for the other model variables; they are not causal effects.",
        transform=ax.transAxes, fontsize=8.5, color="#4B5563", va="top",
    )
    fig.subplots_adjust(left=0.33, right=0.96, top=0.92, bottom=0.28)
    fig.savefig(TIMING_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {TIMING_OUTPUT}")


def main() -> None:
    all_results = pd.read_csv(SUMMARY)
    create_release_timing_plot(all_results)
    results = all_results.copy()
    # keeps the main predictors. English only and release timing has own plot.
    results = results[
        ~results["term"].str.startswith("C(release_month)")
        & ~results["term"].eq("release_year_decade")
        & ~results["term"].str.startswith("C(original_language)")
        | results["term"].eq("C(original_language)[T.en]")
    ].copy()

    # rescale continuous predictors to units that are easier to explain.
    units = {
        "vote_average": (1, "TMDB rating (+1 point)"),
        "number_of_episodes": (10, "Episodes (+10 episodes)"),
        "number_of_seasons": (1, "Seasons (+1 season)"),
    }
    labels = []
    for _, row in results.iterrows():
        term = row["term"]
        if term in units:
            multiplier, label = units[term]
            results.loc[row.name, "coef"] *= multiplier
            results.loc[row.name, "coef lower 95%"] *= multiplier
            results.loc[row.name, "coef upper 95%"] *= multiplier
            labels.append(label)
        elif term == "C(original_language)[T.en]":
            labels.append("Language: English")
        else:
            genre = term.split("[T.", 1)[1].rstrip("]")
            labels.append(f"Genre: {genre}")
    results["label"] = labels
    results["hr"] = results["coef"].apply(lambda x: __import__("math").exp(x))
    results["lo"] = results["coef lower 95%"].apply(lambda x: __import__("math").exp(x))
    results["hi"] = results["coef upper 95%"].apply(lambda x: __import__("math").exp(x))
    results["significant"] = results["p"] < 0.05
    results = results.sort_values("hr", ascending=True).reset_index(drop=True)

    fig_height = max(7, 0.34 * len(results) + 2.3)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y = range(len(results))
    colors = ["#147D64" if sig else "#8A94A6" for sig in results["significant"]]
    for yi, (_, row), color in zip(y, results.iterrows(), colors):
        ax.errorbar(
            row["hr"], yi,
            xerr=[[row["hr"] - row["lo"]], [row["hi"] - row["hr"]]],
            fmt="o", color=color, ecolor=color, capsize=2.5, markersize=5,
            linewidth=1.5,
        )

    ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(list(y), results["label"])
    ax.set_xlabel("Adjusted hazard ratio (95% confidence interval; log scale)")
    ax.set_title("Features associated with faster or slower exit from Netflix Global Top 10", loc="left", pad=16)
    ax.grid(axis="x", which="major", color="#D9DEE7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.01, -0.085,
        "Above 1 = faster exit; below 1 = slower exit. Green: p < .05. "
        "Genre estimates compare with Action & Adventure; English compares with Arabic.",
        transform=ax.transAxes, fontsize=9, color="#4B5563", va="top",
    )
    ax.text(
        0.01, -0.12,
        "Model adjusts for genre, language, rating, episodes, seasons, and release month/year. "
        "Associations are not causal effects.",
        transform=ax.transAxes, fontsize=9, color="#4B5563", va="top",
    )
    fig.subplots_adjust(left=0.33, right=0.96, top=0.94, bottom=0.16)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
