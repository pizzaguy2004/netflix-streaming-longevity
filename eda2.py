import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

wks_g = pd.read_csv('data/all-weeks-global.tsv', sep='\t')
mast_ntflx = pd.read_csv('data/master_netflix_analysis.csv')
ntflx_tmbd = pd.read_csv('data/netflix_tmdb_metadata.csv')


#Kaplan-Meier Survival Analysis
kmf = KaplanMeierFitter()
kmf.fit(durations= mast_ntflx['duration_weeks'], event_observed= mast_ntflx['event_observed'])
kmf.plot_survival_function()
plt.show()

#kmf = KaplanMeierFitter()
#for og_language in mast_ntflx['original_language'].unique():
  #mask = mast_ntflx['original_language'] == og_language
  #kmf.fit(
      #mast_ntflx.loc[mask, 'duration_weeks'],
      #mast_ntflx.loc[mask, 'event_observed'],
      #label=og_language,
  #)
  #kmf.plot_survival_function()
#plt.show()
