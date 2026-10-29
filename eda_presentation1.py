import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

wks_g = pd.read_csv('all-weeks-global.tsv', sep='\t')
mast_ntflx = pd.read_csv('master_netflix_analysis.csv')
ntflx_tmbd = pd.read_csv('netflix_tmdb_metadata.csv')

#merge datasets on show title
merged_df = wks_g.merge(mast_ntflx, on="show_title", how="left").merge(ntflx_tmbd, on = "show_title", how = "left")
print(merged_df)

#filters out rows not having a matching tmdb id
filtered_df = merged_df.dropna(subset=['tmdb_id_y'])
print(filtered_df)



seperated_genres = filtered_df['genres_y'].str.split(', | & ').explode()

counts = seperated_genres.value_counts()

threshold = 5
rare = counts[counts < threshold].index

clean_genres = seperated_genres.replace(rare, 'Other')

final_counts = clean_genres.value_counts().reset_index()

final_counts.columns = ['Genre', 'Count']

#bar plot of specific genre frequencies. Breaks up multiple genre discriptions into individual to make plot cleaner.
plt.figure(figsize=(10,6))
sns.barplot(data=final_counts, x = 'Count', y = 'Genre', palette = 'muted')
plt.title('Frequency of Genre ')
plt.tight_layout()
plt.show()



szn_count = filtered_df['number_of_seasons_y'].value_counts().reset_index()

szn_count.columns = ['# of seasons', 'Count']

plt.figure(figsize = (10, 5))

sns.barplot( 
    data = szn_count, y = 'Count', x = '# of seasons', palette = 'muted'
)
plt.title('Season count')
plt.tight_layout()
plt.show()



ep_count = filtered_df['number_of_episodes_y'].value_counts().reset_index()

ep_count.columns = ['# of episodes', 'Count']

plt.figure(figsize = (10, 5))

sns.barplot( 
    data = ep_count, y = 'Count', x = '# of episodes', palette = 'muted'
)
plt.title('Episode count')
plt.tight_layout()
plt.show()



# chanage the values of the bins if we need to observe a wider range of # of episodes per show title
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, float('inf')]

labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-45)', '(46-50)', '(51+)']

ep_count['Range Group'] = pd.cut(ep_count['Count'], bins=bins, labels=labels, right = True)

range_dist = ep_count['Range Group'].value_counts().reset_index()
range_dist.columns = ['Count Range', '# of eps']

plt.figure(figsize = (10, 5))

sns.barplot( 
    data = range_dist, x = 'Count Range', y = '# of eps', palette = 'muted'
)
plt.title('Distribution of Episodes across frequency ranges')
plt.tight_layout()
plt.show()
