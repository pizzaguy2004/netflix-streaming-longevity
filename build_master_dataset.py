import pandas as pd

# Load datasets
netflix_df = pd.read_csv("data/all-weeks-global.tsv", sep="\t")
tmdb_df = pd.read_csv("data/netflix_tmdb_metadata.csv")
netflix_df["week"] = pd.to_datetime(netflix_df["week"])
exit_gap_weeks = 4

# Filter for TV categories and define first-run survival outcome.
tv_df = netflix_df[netflix_df["category"].str.contains("TV", na=False)].copy()
data_end = tv_df["week"].max()
exit_gap = pd.Timedelta(weeks=exit_gap_weeks)

# Keep existing title-level descriptive fields, which summarize all appearances.
summary_df = (
    tv_df.groupby("show_title")
    .agg(
        total_hours=("weekly_hours_viewed", "sum"),
        peak_rank=("weekly_rank", "min"),
    )
    .reset_index()
)

# One survival record per title, starting at its first chart appearance.
run_rows = []
for title, group in tv_df.groupby("show_title"):
    weeks = sorted(group["week"].drop_duplicates())
    first_week = weeks[0]

    # A gap of at least four weeks ends the first run; later returns are ignored.
    gap_index = next(
        (i for i in range(1, len(weeks)) if weeks[i] - weeks[i - 1] >= exit_gap),
        None,
    )

    if gap_index is not None:
        end_week = weeks[gap_index - 1]
        event_observed = 1
    elif data_end - weeks[-1] >= exit_gap:
        end_week = weeks[-1]
        event_observed = 1
    else:
        # Follow-up ends before a four-week absence confirms an exit.
        end_week = data_end
        event_observed = 0

    run_rows.append(
        {
            "show_title": title,
            "duration_weeks": (end_week - first_week).days / 7 + 1,
            "event_observed": event_observed,
        }
    )

y_df = pd.DataFrame(run_rows).merge(summary_df, on="show_title", how="left")

# Merge outcome (Y) with predictors (X)
master_df = pd.merge(y_df, tmdb_df, on="show_title", how="inner")

# Clean missing values and primary genre
master_df["primary_genre"] = master_df["genres"].fillna("Unknown").apply(
    lambda x: x.split(", ")[0] if x != "Unknown" else "Unknown"
)

# Save master dataset
master_df.to_csv("data/master_netflix_analysis.csv", index=False)
print(f"Master dataset created with {len(master_df)} shows!")
print(master_df.head())
