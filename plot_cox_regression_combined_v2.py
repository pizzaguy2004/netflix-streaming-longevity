# Plots cox regression hazard results for relevant features

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from cox_regression_model import load_model_data


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "cox_regression_summary_v2.csv"
OUTPUT_PATH = ROOT / "cox_forest_combined_v2.png"


def main() -> None:
    results = pd.read_csv(SUMMARY_PATH)
    model_data = load_model_data()
    genre_counts = model_data["genre"].value_counts().to_dict()
    language_counts = model_data["lang"].value_counts().to_dict()

    results = results[
        results["term"].str.startswith(("C(genre)", "C(lang)"))
        | results["term"].isin(["vote_average_z", "log_episodes_z", "number_of_seasons_z"])
    ].copy()

    lang_names = {"ko": "Korean", "es": "Spanish", "ja": "Japanese", "other": "Other language"}
    labels = []
    groups = []
    for term in results["term"]:
        if term.startswith("C(genre)"):
            group = term.split("[T.", 1)[1].rsplit("]", 1)[0]
            labels.append(f"Genre: {group} (n={genre_counts[group]})")
            groups.append("Genre")
        elif term.startswith("C(lang)"):
            lang = term.split("[T.", 1)[1].rsplit("]", 1)[0]
            labels.append(f"{lang_names[lang]} (n={language_counts[lang]}; vs English)")
            groups.append("Other predictors")
        elif term == "vote_average_z":
            rating_sd = model_data["vote_average"].std()
            labels.append(f"TMDB rating (+1 SD = {rating_sd:.1f} points)")
            groups.append("Other predictors")
        elif term == "log_episodes_z":
            labels.append("Log episode count (+1 SD)")
            groups.append("Other predictors")
        else:
            season_sd = model_data["number_of_seasons"].std()
            labels.append(f"Number of seasons (+1 SD = {season_sd:.1f})")
            groups.append("Other predictors")

    results["label"] = labels
    results["group"] = groups
    results["hr"] = results["exp(coef)"]
    results["lo"] = results["exp(coef) lower 95%"]
    results["hi"] = results["exp(coef) upper 95%"]
    results["significant"] = results["p"] < 0.05

    predictors = results[results["group"] == "Other predictors"].sort_values("hr")
    genres = results[results["group"] == "Genre"].sort_values("hr")
    results = pd.concat([predictors, genres], ignore_index=True)

    fig_height = max(8, 0.43 * len(results) + 2.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y = list(range(len(results)))
    colors = ["#14866D" if sig else "#8B95A7" for sig in results["significant"]]
    for yi, (_, row), color in zip(y, results.iterrows(), colors):
        ax.errorbar(
            row["hr"], yi,
            xerr=[[row["hr"] - row["lo"]], [row["hi"] - row["hr"]]],
            fmt="o", color=color, ecolor=color, capsize=2.8, markersize=5.5,
            linewidth=1.6,
        )

    ax.axvline(1, color="#303744", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.47, 4.1)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.4, 2, 3, 4]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "0.7", "1", "1.4", "2", "3", "4"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(y, results["label"])
    ax.invert_yaxis()
    ax.axhline(len(predictors) - 0.5, color="#C4CBD5", linewidth=1)
    ax.set_xlabel("Adjusted hazard ratio (95% CI; log scale)")
    ax.set_title("Model estimates for Top 10 exit", loc="left", pad=30)
    ax.text(
        0.5, 1.015, "← slower exit     faster exit →",
        transform=ax.transAxes, ha="center", va="bottom", color="#4B5563", fontsize=10,
    )
    ax.grid(axis="x", which="major", color="#D9DEE7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.0, -0.10,
        "Genre compares with Drama (n=412); language compares with English (n=571). "
        "Green: p < .05. Model n=1,209 (1,180 exits; 29 censored); adjusts for release month and year.",
        transform=ax.transAxes, fontsize=8.8, color="#4B5563", va="top",
    )
    ax.text(
        0.0, -0.14,
        "Continuous effects are per +1 SD; episode count is log-transformed. Estimates are associations, not causal effects.",
        transform=ax.transAxes, fontsize=8.8, color="#4B5563", va="top",
    )
    fig.subplots_adjust(left=0.35, right=0.97, top=0.92, bottom=0.16)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
