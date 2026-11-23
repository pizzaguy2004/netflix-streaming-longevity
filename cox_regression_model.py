from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter


ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "data" / "master_netflix_analysis.csv"
SUMMARY_PATH = ROOT / "cox_regression_summary.csv"


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

    formula = (
        "C(primary_genre) + C(original_language) + vote_average + "
        "number_of_seasons + number_of_episodes + C(release_month) + "
        "release_year_decade"
    )

    cph = CoxPHFitter()
    cph.fit(
        model_data,
        duration_col="duration_weeks",
        event_col="event_observed",
        formula=formula,
    )

    cph.print_summary()
    cph.summary.to_csv(SUMMARY_PATH, index_label="term")
    print(f"Saved model summary to {SUMMARY_PATH}")
    print(
        f"Model sample: {len(model_data)} shows; "
        f"{int(model_data['event_observed'].sum())} confirmed exits; "
        f"{int((model_data['event_observed'] == 0).sum())} censored."
    )


if __name__ == "__main__":
    main()
