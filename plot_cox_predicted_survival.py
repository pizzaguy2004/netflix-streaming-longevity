
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cox_regression_model import fit_model

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cox_predicted_survival_v2.png"

BASE = dict(number_of_seasons=1, number_of_episodes=10, release_month="1", release_year_decade=0.5)
PROFILES = {
    "High-rated Korean drama (8.5)":      dict(genre="Drama", lang="ko", vote_average=8.5, **BASE),
    "Low-rated Korean drama (5.5)":       dict(genre="Drama", lang="ko", vote_average=5.5, **BASE),
    "Low-rated English reality show (5.5)": dict(genre="Reality", lang="en", vote_average=5.5, **BASE),
}
COLORS = ["#147D64", "#D9A441", "#C75B39"]


def main() -> None:
    cph, df = fit_model()
    prof = pd.DataFrame(PROFILES).T.reset_index(names="profile")
    for col in ["number_of_seasons", "number_of_episodes", "vote_average", "release_year_decade"]:
        prof[col] = prof[col].astype(float)
    prof["log_episodes"] = np.log(prof["number_of_episodes"])
    # standardize with the SAME means/SDs used to fit the model
    for col in ["vote_average", "log_episodes", "number_of_seasons"]:
        prof[col + "_z"] = (prof[col] - df[col].mean()) / df[col].std()
    prof["genre"] = pd.Categorical(prof["genre"], categories=df["genre"].cat.categories)
    prof["lang"] = pd.Categorical(prof["lang"], categories=df["lang"].cat.categories)
    prof = prof.set_index("profile")

    survival = cph.predict_survival_function(prof)
    medians = cph.predict_median(prof)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, curve), color in zip(survival.items(), COLORS):
        med = medians.loc[name]
        ax.step(curve.index, curve.values, where="post", color=color, linewidth=2.5,
                label=f"{name}: median {med:.0f} wk")
        ax.axvline(med, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axhline(0.5, color="#697386", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Weeks since first Global Top 10 appearance")
    ax.set_ylabel("Predicted probability of still being in the Top 10")
    ax.set_title("Predicted Top 10 survival for three illustrative shows", loc="left", pad=14)
    ax.grid(color="#E1E5EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.text(0.01, 0.01,
             "All profiles: 1 season, 10 episodes, January release, 2025. Medians are whole weeks "
             "(durations are integer weeks). Model predictions, not causal estimates.",
             fontsize=8.5, color="#4B5563")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.14)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    plt.close(fig)

    print(f"Saved {OUTPUT}")
    for name, med in medians.items():
        print(f"{name}: predicted median = {med:.1f} weeks")


if __name__ == "__main__":
    main()
