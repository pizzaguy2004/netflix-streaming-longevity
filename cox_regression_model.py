# Cox model 
# includes cleaned zeros, english/drama reference levels, collapsed small categories, log-episodes, standardized continuous predictors, plus diagnostics

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "data" / "master_netflix_analysis.csv"
SUMMARY_PATH = ROOT / "cox_regression_summary_v2.csv"
MIN_GENRE_N = 20          # genres with fewer shows are pooled into "Other"
TOP_LANGS = ["en", "ko", "es", "ja"]

COLS = ["duration_weeks", "event_observed", "genre", "lang", "vote_average_z",
        "log_episodes_z", "number_of_seasons_z", "release_month", "release_year_decade"]
FORMULA = ("C(genre) + C(lang) + vote_average_z + log_episodes_z + "
           "number_of_seasons_z + C(release_month) + release_year_decade")


def load_model_data() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH)
    input_n = len(df)
    release = pd.to_datetime(df["first_air_date"], errors="coerce")
    df = df.loc[release.notna()].copy()
    release = release.loc[df.index]
    df["release_month"] = release.dt.month.astype(str)
    df["release_year_decade"] = (release.dt.year - 2020) / 10

    # 0 is a placeholder for missing TMDB data, not a real value.
    for col in ["vote_average", "number_of_episodes", "number_of_seasons"]:
        df.loc[df[col] == 0, col] = np.nan
    df = df.dropna(subset=["vote_average", "number_of_episodes", "number_of_seasons"])
    metadata_filtered_n = len(df)
    censored_kept_n = int((df["event_observed"] == 0).sum())

    # Language: English is the reference; everything beyond the top 4 -> "other".
    df["lang"] = np.where(df["original_language"].isin(TOP_LANGS), df["original_language"], "other")
    df["lang"] = pd.Categorical(df["lang"], categories=TOP_LANGS + ["other"])

    # Genre: Drama (largest group) is the reference; tiny genres pooled.
    counts = df["primary_genre"].value_counts()
    tiny = counts[counts < MIN_GENRE_N].index
    df["genre"] = df["primary_genre"].where(~df["primary_genre"].isin(tiny), "Other (small genres)")
    levels = ["Drama"] + sorted(g for g in df["genre"].unique() if g != "Drama")
    df["genre"] = pd.Categorical(df["genre"], categories=levels)

    # Episodes are heavily right-skewed -> log; then standardize (per 1 SD).
    df["log_episodes"] = np.log(df["number_of_episodes"])
    for col in ["vote_average", "log_episodes", "number_of_seasons"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    df.attrs["input_n"] = input_n
    df.attrs["metadata_dropped_n"] = input_n - metadata_filtered_n
    df.attrs["censored_kept_n"] = censored_kept_n
    return df


def fit_model():
    """Return (fitted CoxPHFitter, modeling DataFrame). Shared by all plot scripts."""
    df = load_model_data()
    cph = CoxPHFitter()  # ties handled with the Efron approximation (lifelines default)
    cph.fit(df[COLS], duration_col="duration_weeks", event_col="event_observed", formula=FORMULA)
    return cph, df


def main() -> None:
    cph, df = fit_model()
    cols = COLS
    cph.print_summary()
    cph.summary.to_csv(SUMMARY_PATH, index_label="term")

    print(f"\nn = {len(df)} shows retained; {int(df['event_observed'].sum())} confirmed exits; "
          f"{int((df['event_observed'] == 0).sum())} right-censored (kept in model)")
    print(f"Dropped overall: {df.attrs['input_n'] - len(df)} of {df.attrs['input_n']} shows "
          f"({df.attrs['metadata_dropped_n']} for missing release date and/or placeholder predictors)")
    print(f"Concordance index (in-sample): {cph.concordance_index_:.3f}")

    ph = proportional_hazard_test(cph, df[cols], time_transform="rank").summary
    print(f"\nSchoenfeld PH test: {int((ph['p'] < 0.05).sum())} of {len(ph)} terms with p < .05")
    print(ph.loc[ph["p"] < 0.05, ["test_statistic", "p"]].round(4))

    X = df[["vote_average", "number_of_seasons", "log_episodes"]].assign(const=1.0)
    vif = {c: round(variance_inflation_factor(X.values, i), 2)
           for i, c in enumerate(X.columns) if c != "const"}
    print(f"\nVIF: {vif}")
    print(f"corr(seasons, log episodes) = {df['number_of_seasons'].corr(df['log_episodes']):.2f}")
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
