# Forest plots for the Cox model (run cox_regression_model.py first).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cox_regression_model import load_model_data

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "cox_regression_summary_v2.csv"
GREEN, GRAY = "#147D64", "#8A94A6"
MONTHS = {"2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun", "7": "Jul",
          "8": "Aug", "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"}


def forest(rows, title, footnote, out, xlim, ticks, figsize=(9, 5.5)):
    """rows: list of (label, hr, lo, hi, p) -> forest plot on a log axis."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (_, hr, lo, hi, p) in enumerate(rows):
        color = GREEN if p < 0.05 else GRAY
        ax.errorbar(hr, i, xerr=[[hr - lo], [hi - hr]], fmt="o", color=color,
                    ecolor=color, capsize=3, markersize=6, linewidth=1.8)
    ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Adjusted hazard ratio (95% CI, log scale)")
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.8)
    ax.set_axisbelow(True)
    # direction cues just above the plot, either side of HR = 1
    tr = ax.get_xaxis_transform()
    ax.text(0.97, 1.02, "\u2190 slower exit", ha="right", va="bottom", transform=tr,
            fontsize=10, color="#4B5563")
    ax.text(1.03, 1.02, "faster exit \u2192", ha="left", va="bottom", transform=tr,
            fontsize=10, color="#4B5563")
    ax.set_title(title, loc="left", pad=28, fontsize=13)
    fig.text(0.01, 0.005, footnote, fontsize=8.5, color="#4B5563", va="bottom")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.86, bottom=0.2)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    s = pd.read_csv(SUMMARY).set_index("term")
    df = load_model_data()
    n_genre = df["genre"].value_counts()

    def row(term, label):
        r = s.loc[term]
        return (label, r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"], r["p"])

    # --- genres (reference = Drama), sorted by hazard ratio, n in label
    g = [row(t, f"{t.split('[T.')[1].rstrip(']')} (n={n_genre[t.split('[T.')[1].rstrip(']')]})")
         for t in s.index if t.startswith("C(genre)")]
    g.sort(key=lambda r: -r[1])
    forest(g, "Genre and Top 10 exit",
           f"Compared with Drama (n={n_genre['Drama']}). Green: p < .05. Genres with < 20 shows pooled "
           "into 'Other'. Adjusted for all model variables; associations, not causal effects.",
           ROOT / "cox_forest_genre_v2.png", (0.4, 4), [0.5, 1, 2, 4])

    # --- predictors (continuous per 1 SD, language vs English)
    sd = {"vote_average_z": df["vote_average"].std(), "log_episodes_z": df["log_episodes"].std(),
          "number_of_seasons_z": df["number_of_seasons"].std()}
    p = [
        row("vote_average_z", f"TMDB rating (+1 SD = {sd['vote_average_z']:.1f} pts)"),
        row("log_episodes_z", "Log episodes (+1 SD)"),
        row("number_of_seasons_z", f"Seasons (+1 SD = {sd['number_of_seasons_z']:.1f})"),
        row("C(lang)[T.ko]", "Korean (vs English)"),
        row("C(lang)[T.es]", "Spanish (vs English)"),
        row("C(lang)[T.ja]", "Japanese (vs English)"),
        row("C(lang)[T.other]", "Other language (vs English)"),
    ]
    forest(p, "Ratings, length, and language and Top 10 exit",
           "Continuous predictors are standardized (per 1 SD); episodes are log-transformed. "
           "Green: p < .05. Adjusted for genre and release timing; not causal.",
           ROOT / "cox_forest_predictors_v2.png", (0.5, 1.8), [0.5, 0.7, 1, 1.4], figsize=(9, 4.8))

    # --- timing appendix
    t = [row(f"C(release_month)[T.{m}]", f"{name} (vs Jan)") for m, name in MONTHS.items()]
    t.append(row("release_year_decade", "Release year (+10 years)"))
    forest(t, "Release timing and Top 10 exit",
           "Months compare with January. Green: p < .05. Controls only; not causal.",
           ROOT / "cox_forest_timing_v2.png", (0.4, 2), [0.5, 1, 2], figsize=(9, 6.5))


if __name__ == "__main__":
    main()
