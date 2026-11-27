# Plot Cox model survival curves for two illustrative show profiles.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter


ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "data" / "master_netflix_analysis.csv"
OUTPUT_PATH = ROOT / "cox_predicted_survival_profiles.png"

FORMULA = (
    "C(primary_genre) + C(original_language) + vote_average + "
    "number_of_seasons + number_of_episodes + C(release_month) + "
    "release_year_decade"
)


def main() -> None:
    master = pd.read_csv(MASTER_PATH)
    release_date = pd.to_datetime(master["first_air_date"], errors="coerce")
    master["release_month"] = release_date.dt.month.astype("Int64").astype("string")
    master["release_year_decade"] = (release_date.dt.year - 2020) / 10

    model_data = master.loc[release_date.notna(), [
        "duration_weeks",
        "event_observed",
        "primary_genre",
        "original_language",
        "vote_average",
        "number_of_seasons",
        "number_of_episodes",
        "release_month",
        "release_year_decade",
    ]].copy()

    cph = CoxPHFitter()
    cph.fit(
        model_data,
        duration_col="duration_weeks",
        event_col="event_observed",
        formula=FORMULA,
    )

    # Holds duration related metadata and release timing fixed so the curves compare two tv show profiles with the same 10 episodes and 1 season.
    profiles = pd.DataFrame([
        {
            "profile": "Higher-rated Korean drama",
            "primary_genre": "Drama",
            "original_language": "ko",
            "vote_average": 8.5,
            "number_of_seasons": 1,
            "number_of_episodes": 10,
            "release_month": "1",
            "release_year_decade": 0.5,
        },
        {
            "profile": "Lower-rated English reality show",
            "primary_genre": "Reality",
            "original_language": "en",
            "vote_average": 5.5,
            "number_of_seasons": 1,
            "number_of_episodes": 10,
            "release_month": "1",
            "release_year_decade": 0.5,
        },
    ]).set_index("profile")

    survival = cph.predict_survival_function(profiles)
    medians = cph.predict_median(profiles)

    colors = ["#147D64", "#C75B39"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, curve), color in zip(survival.items(), colors):
        median = medians.loc[name]
        median_text = "not reached" if pd.isna(median) or median == float("inf") else f"{median:.1f} weeks"
        ax.step(
            curve.index,
            curve.values,
            where="post",
            color=color,
            linewidth=2.5,
            label=f"{name} (median: {median_text})",
        )
        if pd.notna(median) and median != float("inf"):
            ax.axvline(median, color=color, linestyle=":", linewidth=1.3, alpha=0.8)

    ax.axhline(0.5, color="#697386", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Weeks since first Global Top 10 appearance")
    ax.set_ylabel("Predicted probability of remaining in the Top 10")
    ax.set_title("Illustrative Cox model survival curves", loc="left", pad=14)
    ax.grid(axis="both", color="#E1E5EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    takeaway = (
        "Illustrative model medians: 6 vs. 2 weeks. Profiles differ in rating, genre, "
        "and language, so this is not an isolated rating effect or a causal estimate."
    )
    ax.text(
        0.0, -0.18, takeaway,
        transform=ax.transAxes,
        fontsize=9,
        color="#4B5563",
        va="top",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.24)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    for name, median in medians.items():
        value = "not reached" if pd.isna(median) or median == float("inf") else f"{median:.1f} weeks"
        print(f"{name}: predicted median = {value}")


if __name__ == "__main__":
    main()
