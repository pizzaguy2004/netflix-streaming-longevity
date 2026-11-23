import pandas as pd
import matplotlib.pyplot as plt

master = pd.read_csv("master_netflix_analysis.csv")

stats = master.describe()
summary = master.info()

# drops non-relevant columns from summary stats dataframe
stats = stats.drop(
    columns=["tmdb_id", "event_observed", "total_hours", "peak_rank", "vote_count"]
)
stats = stats.drop(index="count")
stats = stats.rename(
    columns={
        "duration_weeks": "Chart duration (weeks)",
        "vote_average": "TMDB rating (out of 10)",
        "number_of_episodes": "Number of episodes",
        "number_of_seasons": "Number of seasons",
    }
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=stats.values, 
                 rowLabels=stats.index, 
                 colLabels=stats.columns, 
                 cellLoc='center', 
                 loc='center')

plt.savefig('dataset_summary_table.png', bbox_inches='tight', dpi=300)

