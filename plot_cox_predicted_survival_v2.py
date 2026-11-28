from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cox_regression_model import fit_model


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "cox_predicted_survival_v2.png"


def main() -> None:
    cph, model_data = fit_model()

    rating_mean = model_data["vote_average"].mean()
    rating_sd = model_data["vote_average"].std()
    log_episode_mean = model_data["log_episodes"].mean()
    log_episode_sd = model_data["log_episodes"].std()
    season_mean = model_data["number_of_seasons"].mean()
    season_sd = model_data["number_of_seasons"].std()

    raw_profiles = [
        {"profile": "High-rated Korean drama (8.5)",
            "genre": "Drama", "lang": "ko", "rating": 8.5,
        },
        {"profile": "Low-rated Korean drama (5.5)",
            "genre": "Drama", "lang": "ko", "rating": 5.5,
        },
        {"profile": "Low-rated English reality show (5.5)",
            "genre": "Reality", "lang": "en", "rating": 5.5,
        },
    ]

    profiles = pd.DataFrame([
        {"genre": row["genre"], "lang": row["lang"], "vote_average_z": (row["rating"] - rating_mean) / rating_sd,
         "log_episodes_z": (np.log(10) - log_episode_mean) / log_episode_sd, "number_of_seasons_z": (1 - season_mean) / season_sd,
         "release_month": "1", "release_year_decade": 0.5,}
        for row in raw_profiles
    ], index=[row["profile"] for row in raw_profiles])
    profiles["genre"] = pd.Categorical(profiles["genre"], categories=model_data["genre"].cat.categories)
    profiles["lang"] = pd.Categorical(profiles["lang"], categories=model_data["lang"].cat.categories)

    survival = cph.predict_survival_function(profiles)
    medians = cph.predict_median(profiles)
    colors = ["#13866D", "#D9A441", "#C65D3B"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, curve), color in zip(survival.items(), colors):
        median = medians.loc[name]
        median_text = "not reached" if pd.isna(median) or median == float("inf") else f"{median:.0f} wk"
        ax.step(
            curve.index,
            curve.values,
            where="post",
            color=color,
            linewidth=2.4,
            label=f"{name}: median {median_text}",
        )
        if pd.notna(median) and median != float("inf"):
            ax.axvline(median, color=color, linestyle=":", linewidth=1.2, alpha=0.8)

    ax.axhline(0.5, color="#697386", linestyle="--", linewidth=1)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Weeks since first Global Top 10 appearance")
    ax.set_ylabel("Predicted probability of remaining in the Top 10")
    ax.set_title("Predicted Top 10 survival for illustrative shows", loc="left", pad=14)
    ax.grid(axis="both", color="#E1E5EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8.7)
    ax.text(0.0, -0.17,
        "All profiles: 1 season, 10 episodes, January 2025 release. "
        "Curves use the current v2 Cox model (1,209 shows; 29 right-censored).",
        transform=ax.transAxes, fontsize=8.8, color="#4B5563", va="top",
    )
    ax.text(0.0, -0.21,
        "The profiles differ in rating, genre, and/or language; contrasts are illustrative, not causal.",
        transform=ax.transAxes, fontsize=8.8, color="#4B5563", va="top",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.24)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    for name, median in medians.items():
        value = "not reached" if pd.isna(median) or median == float("inf") else f"{median:.0f} weeks"
        print(f"{name}: predicted median = {value}")


if __name__ == "__main__":
    main()
