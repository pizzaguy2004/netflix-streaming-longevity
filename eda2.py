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
original_language_avg_longevity = dict(sorted(original_language_avg_longevity.items()))
print(original_language_avg_longevity)

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
