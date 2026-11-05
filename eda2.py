import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import numpy as np

wks_g = pd.read_csv('data/all-weeks-global.tsv', sep='\t')
mast_ntflx = pd.read_csv('data/master_netflix_analysis.csv')
ntflx_tmbd = pd.read_csv('data/netflix_tmdb_metadata.csv')



#Kaplan-Meier Survival Analysis Plot
kmf = KaplanMeierFitter()
kmf.fit(durations= mast_ntflx['duration_weeks'], event_observed= mast_ntflx['event_observed'])
kmf.plot_survival_function()
#plt.show()
plt.close()



#Kaplan-Meier Survival Analysis by Original Language
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
plt.close()



#Number of shows by first air date month
month_counts = {}
for i in mast_ntflx['first_air_date']:
    if i is None or isinstance(i, float):
        continue
    month = i[5:7]
    if month in month_counts:
        month_counts[month] += 1
    else:
        month_counts[month] = 1
print(month_counts)
sorted_month_counts = dict(sorted(month_counts.items()))
print(sorted_month_counts)

plt.figure(figsize=(10, 6))
plt.bar(sorted_month_counts.keys(), sorted_month_counts.values())
plt.xlabel('Month')
plt.ylabel('Number of Shows')
plt.title('Number of Shows by First Air Date Month')
#plt.show()
plt.close()


#Average show longevity by month
month_longevity = {}
for i, j in zip(mast_ntflx['first_air_date'], mast_ntflx['duration_weeks']) :
    if i is None or isinstance(i, float):
        continue
    month = i[5:7]
    if month in month_longevity:
        month_longevity[month] += j
    else:
        month_longevity[month] = j


average_longevity = {month: total / month_counts[month] for month, total in month_longevity.items()}
average_longevity = dict(sorted(average_longevity.items()))
print(average_longevity)

plt.figure(figsize=(10, 6))
plt.bar(average_longevity.keys(), average_longevity.values())
plt.xlabel('Month')
plt.ylabel('Average Longevity (Weeks)')
plt.title('Average Show Longevity by First Air Date Month')
#plt.show()
plt.close()



#Plot of original language and average longevity
original_language_count = {}
original_language_avg_longevity = {}
for i, j in zip(mast_ntflx['original_language'], mast_ntflx['duration_weeks']):
    if i is None or isinstance(i, float):
        continue
    if i in original_language_count:
        original_language_count[i] += 1
        original_language_avg_longevity[i] += j
    else:
        original_language_count[i] = 1
        original_language_avg_longevity[i] = j

original_language_avg_longevity = {lang: total / original_language_count[lang] for lang, total in original_language_avg_longevity.items()}
original_language_avg_longevity = dict(sorted(original_language_avg_longevity.items(), key=lambda item: item[1], reverse=True))
print(original_language_avg_longevity)

original_language_count = dict(sorted(original_language_count.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.bar(original_language_count.keys(), original_language_count.values())
plt.xlabel('Original Language')
plt.ylabel('Number of Shows')
plt.title('Number of Shows by Original Language')
#plt.show()
plt.close()



plt.figure(figsize=(10, 6))
plt.bar(original_language_avg_longevity.keys(), original_language_avg_longevity.values())
plt.xlabel('Original Language')
plt.ylabel('Average Longevity (Weeks)')
plt.title('Average Show Longevity by Original Language')
#plt.show()
plt.close()




#Average show rating by longevity weeks
longevity_weeks_count = {}
avg_longevity_rating = {}
for i, j in zip(mast_ntflx['duration_weeks'], mast_ntflx['vote_average']):
    if i in longevity_weeks_count:
        longevity_weeks_count[i] += 1
        avg_longevity_rating[i] += j
    else:
        longevity_weeks_count[i] = 1
        avg_longevity_rating[i] = j
avg_longevity_rating = {week: total / longevity_weeks_count[week] for week, total in avg_longevity_rating.items()}
avg_longevity_rating = dict(sorted(avg_longevity_rating.items()))
longevity_count = dict(sorted(longevity_weeks_count.items()))
print(avg_longevity_rating)
print(longevity_count)
   #Longevity by number of shows
plt.figure(figsize=(10, 6))
plt.bar(longevity_count.keys(), longevity_count.values())
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Number of Shows')
plt.title('Number of Shows by Longevity (Weeks)')
#plt.show()
plt.close()
    #Average show rating by longevity weeks
plt.figure(figsize=(10, 6))
plt.plot(avg_longevity_rating.keys(), avg_longevity_rating.values())
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Average Rating')
plt.title('Average Show Rating by Longevity (Weeks)')
#plt.show()
plt.close()




#Plot of peak rank by longevity weeks

peak_rank_longevity = {}
peak_rank_count = {}
for i, j in zip(mast_ntflx['duration_weeks'], mast_ntflx['peak_rank']):
    if i in peak_rank_longevity:
        peak_rank_longevity[i] += j
        peak_rank_count[i] += 1
    else:
        peak_rank_longevity[i] = j
        peak_rank_count[i] = 1

peak_rank_longevity = {week: peak_rank_longevity[week] / peak_rank_count[week] for week in peak_rank_longevity}
peak_rank_longevity = dict(sorted(peak_rank_longevity.items()))
print(peak_rank_longevity)

plt.figure(figsize=(10, 6))
plt.plot(peak_rank_longevity.keys(), peak_rank_longevity.values())
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Average Peak Rank')
plt.title('Average Peak Rank by Longevity (Weeks)')
plt.gca().invert_yaxis()
#plt.show()
plt.close()


#Plot of shows released by month and year
shows_by_month_year = {}    
for i in mast_ntflx['first_air_date']:
    if i is None or isinstance(i, float):
        continue
    month = i[5:7]
    year = i[:4]
    month_year = f"{year}-{month}"
    if month_year in shows_by_month_year and int(year) >= 2016:
        shows_by_month_year[month_year] += 1
    elif int(year) >= 2016:
        shows_by_month_year[month_year] = 1
    else:
        continue
show_by_month_year = dict(sorted(shows_by_month_year.items()))

plt.figure(figsize=(10, 6))
keys = list(show_by_month_year.keys())
values = list(show_by_month_year.values())
plt.plot(keys, values)
tick_positions = [i for i, k in enumerate(keys) if k.endswith('-01')]
tick_labels = [keys[i][:4] for i in tick_positions]
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
plt.xlabel('Year')
plt.ylabel('Number of Shows')
plt.title('Number of Shows Released by Month and Year')
plt.tight_layout()
#plt.show()
plt.close()


#Correlation Heatmap of Relavant Numeric Variables
numeric_cols = [
    'duration_weeks',
    'total_hours',
    'peak_rank',
    'vote_average',
    'vote_count',
    'number_of_seasons', 
    'number_of_episodes',
    ]

corr = mast_ntflx[numeric_cols].corr(method='pearson')  # or 'spearman' for rank order

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr,
    annot=True,  
    fmt='.2f',  
    cmap='coolwarm',  
    vmin=-1,
    vmax=1,
    square=True, 
    linewidths=0.5, 
    cbar_kws={'shrink': 0.8},
)

plt.title('Correlation Matrix of Numeric Features', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
#plt.show()
plt.close()



#Scatter plot of duration and hours 

plt.figure(figsize=(10, 6))
plt.scatter(mast_ntflx['duration_weeks'], mast_ntflx['number_of_episodes'])
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Number of Episodes')
plt.title('Number of Episodes vs Longevity (Weeks)')
#plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(mast_ntflx['duration_weeks'], mast_ntflx['vote_average'])
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Vote Average')
plt.title('Vote Average vs Longevity (Weeks)')
#plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(mast_ntflx['duration_weeks'], mast_ntflx['number_of_seasons'])
plt.xlabel('Longevity (Weeks)')
plt.ylabel('Number of Seasons')
plt.title('Number of Seasons vs Longevity (Weeks)')
#plt.show()
plt.close()

#Bar plot of average longevity for multi vs single season shows
season_longevity = {'single': 0, 'multi': 0}
season_count = {'single': 0, 'multi': 0}
for i, j in zip(mast_ntflx['duration_weeks'], mast_ntflx['number_of_seasons']):
    if j == 1:
        season_longevity['single'] += i
        season_count['single'] += 1
    else:
        season_longevity['multi'] += i
        season_count['multi'] += 1

season_longevity = {season: season_longevity[season] / season_count[season] for season in season_longevity}
print(season_longevity)

plt.figure(figsize=(10, 6))
plt.bar(season_longevity.keys(), season_longevity.values())
plt.xlabel('Season Type')
plt.ylabel('Average Longevity (Weeks)')
plt.title('Average Longevity by Season Type')
#plt.show()
plt.close()
