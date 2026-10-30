# initial cox regression model test to determine validity of research question and data
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

master = pd.read_csv("master_netflix_analysis.csv")

master = master.drop(columns = ["show_title", "tmdb_id", "genres", "total_hours", "peak_rank"])

cph = CoxPHFitter()

formula = (
    "C(primary_genre) + C(original_language) + "
    "vote_average + vote_count + number_of_episodes + number_of_seasons"
)

# fits data with cox regression. max_weeks is response variable and event observed determines if show left top 10 before last date in dataset
cph.fit(master, duration_col='max_weeks', event_col = "event_observed", formula = formula)

# summary statistics that give significance levels for each variable and whether tv shows with that specific category have a signifiantly different longevity performance in the netflix top 10
cph.print_summary()

# creates plot showing impact of each variable
cph.plot()
plt.show()