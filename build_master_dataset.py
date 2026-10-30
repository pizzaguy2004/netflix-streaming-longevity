import pandas as pd

# 1. Load datasets
netflix_df = pd.read_csv("data/all-weeks-global.tsv", sep="\t")
tmdb_df = pd.read_csv("data/netflix_tmdb_metadata.csv")
netflix_df["week"] = pd.to_datetime(netflix_df["week"])
exit_gap_weeks = 4

# 2. Filter for TV categories and calculate Y (chart longevity)
tv_df = netflix_df[netflix_df["category"].str.contains("TV", na=False)]
y_df = (
    tv_df
    .groupby("show_title")
    .agg(
        max_weeks=("cumulative_weeks_in_top_10", "max"),
        total_hours=("weekly_hours_viewed", "sum"),
        peak_rank=("weekly_rank", "min"),
        last_chart_week=("week", "max"),
    )
    .reset_index()
)
y_df["event_observed"] = (
    (tv_df["week"].max() - y_df["last_chart_week"]).dt.days
    >= exit_gap_weeks * 7
).astype(int)
y_df = y_df.drop(columns="last_chart_week")

# 3. Merge outcome (Y) with predictors (X)
master_df = pd.merge(y_df, tmdb_df, on="show_title", how="inner")

# 4. Clean missing values and primary genre
master_df["primary_genre"] = master_df["genres"].fillna("Unknown").apply(
    lambda x: x.split(", ")[0] if x != "Unknown" else "Unknown"
)

# Save master dataset
master_df.to_csv("data/master_netflix_analysis.csv", index=False)
print(f"Master dataset created with {len(master_df)} shows!")
print(master_df.head())
