import pandas as pd
import matplotlib.pyplot as plt

master = pd.read_csv("master_netflix_analysis.csv")

stats = master.describe()
summary = master.info()

# drops non-relevant columns from summary stats dataframe
stats = stats.drop(
    columns=["number_of_episodes", "tmdb_id", "event_observed"]
)
stats = stats.drop(index="count")

# Use presentation-friendly labels and units.
stats["total_hours"] = stats["total_hours"] / 1_000_000
stats = stats.rename(
    index={"50%": "Median"},
    columns={
        "duration_weeks": "First-run duration (weeks)",
        "total_hours": "Total hours viewed (millions)",
        "peak_rank": "Best chart rank",
        "vote_average": "TMDB rating",
        "vote_count": "TMDB vote count",
        "number_of_seasons": "TMDB seasons (current)",
    },
).round(1)
stats[["TMDB vote count", "TMDB seasons (current)"]] = stats[
    ["TMDB vote count", "TMDB seasons (current)"]
].round(0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=stats.values, 
                 rowLabels=stats.index, 
                 colLabels=stats.columns, 
                 cellLoc='center', 
                 loc='center')

plt.savefig('dataset_summary_table.png', bbox_inches='tight', dpi=300)

