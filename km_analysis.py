import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test

wks_g = pd.read_csv('data/all-weeks-global.tsv', sep='\t')
mast_ntflx = pd.read_csv('data/master_netflix_analysis.csv')
ntflx_tmbd = pd.read_csv('data/netflix_tmdb_metadata.csv')

dupes = mast_ntflx[mast_ntflx.duplicated(subset=['tmdb_id'])]
print(dupes[['tmdb_id', 'show_title', 'first_air_date']])

# Duplicates found in the dataset:
#1: DOC vs Doc (242)
#2: Jake Paul vs Mike Tyson Premier vs Fight (220)
#3: Love Never Lies vs Love Never Lies Poland (561)
#4: Money Heist vs Heist (Same exact show and row info) (409)
#5: Muscles & Mayhem should be 228550 (641)
#6: Physical 100 vs Physical: 100 Italy (Same) (727)
#7: S&X has same columns as Grey's Anatomy, I suggest removing it (793)
#8: JoJo's Bizarre Adventure same show, different name (485)
#9: The Glory vs Glory (Same show, different name spelling) Suggest removing Glory (377)
#10: The Seven Deadly Sins, same show, different name spelling (1051)

#Removing duplicates
df = mast_ntflx.drop([240, 218, 559, 407, 725, 791, 483, 375, 1049]).copy()
df.at[639, 'tmdb_id'] = 228550
df = df.reset_index(drop=True)

# 1. Seperating Data Based on Language (English vs Korean)
enko = df[df['original_language'].isin(['en', 'ko'])].copy()

en = enko[enko['original_language'] == 'en']
ko = enko[enko['original_language'] == 'ko']

# 2. Fit Kaplan-Meier models
kmf_en = KaplanMeierFitter().fit(
    en['duration_weeks'],
    event_observed=en['event_observed'],
    label='English (en, n=585)',
)
kmf_ko = KaplanMeierFitter().fit(
    ko['duration_weeks'],
    event_observed=ko['event_observed'],
    label='Korean (ko, n=159)',
)
# 3. Log Rank test
lr = logrank_test(
    en['duration_weeks'],
    ko['duration_weeks'],
    event_observed_A=en['event_observed'],
    event_observed_B=ko['event_observed'],
)
wilcoxon = logrank_test(
    en['duration_weeks'],
    ko['duration_weeks'],
    event_observed_A=en['event_observed'],
    event_observed_B=ko['event_observed'],
    weightings='wilcoxon',
)

print(f'Log-Rank: chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.2e}')
print(
    f'Wilcoxon: chi2 = {wilcoxon.test_statistic:.2f}, p ='
    f' {wilcoxon.p_value:.2e}'
)

# 4. Plot survival curves with 95% CI
fig, ax = plt.subplots(figsize=(10, 6))
kmf_en.plot_survival_function(ax=ax, color='#1f77b4', linewidth=2.2)
kmf_ko.plot_survival_function(ax=ax, color='#d62728', linewidth=2.2)

ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.2)
ax.set_title(
    'Kaplan-Meier Survival Curves: English vs. Korean Longevity',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Duration in Netflix Global Top 10 (Weeks)', fontsize=11)
ax.set_ylabel('Survival Probability S(t)', fontsize=11)
ax.set_xlim(0, 22)
ax.grid(True, linestyle='--', alpha=0.5)

add_at_risk_counts(kmf_en, kmf_ko, ax=ax)
plt.tight_layout()
plt.show()
plt.close()


# 1. Seperating Data Based on Single vs Multi-Season
multi = df[df['number_of_seasons'] > 1]
single = df[df['number_of_seasons'] == 1]

# 2. Fit Kaplan-Meier models
kmf_en = KaplanMeierFitter().fit(
    single['duration_weeks'],
    event_observed=single['event_observed'],
    label=f'Single-Season (n={len(single)})',
)
kmf_ko = KaplanMeierFitter().fit(
    multi['duration_weeks'],
    event_observed=multi['event_observed'],
    label=f'Multi-Season (n={len(multi)})',
)
# 3. Log Rank test
lr = logrank_test(
    single['duration_weeks'],
    multi['duration_weeks'],
    event_observed_A=single['event_observed'],
    event_observed_B=multi['event_observed'],
)
wilcoxon = logrank_test(
    single['duration_weeks'],
    multi['duration_weeks'],
    event_observed_A=single['event_observed'],
    event_observed_B=multi['event_observed'],
    weightings='wilcoxon',
)

print(f'Log-Rank: chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.2e}')
print(
    f'Wilcoxon: chi2 = {wilcoxon.test_statistic:.2f}, p ='
    f' {wilcoxon.p_value:.2e}'
)

# 4. Plot survival curves with 95% CI
fig, ax = plt.subplots(figsize=(10, 6))
kmf_en.plot_survival_function(ax=ax, color='#1f77b4', linewidth=2.2)
kmf_ko.plot_survival_function(ax=ax, color='#d62728', linewidth=2.2)

ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.2)
ax.set_title(
    'Kaplan-Meier Survival Curves: Single vs Multi-Season Longevity',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Duration in Netflix Global Top 10 (Weeks)', fontsize=11)
ax.set_ylabel('Survival Probability S(t)', fontsize=11)
ax.set_xlim(0, 22)
ax.grid(True, linestyle='--', alpha=0.5)

add_at_risk_counts(kmf_en, kmf_ko, ax=ax)
plt.tight_layout()
plt.show()
plt.close()


#Multivariate Log-Rank Test For Genre

# 1. Initial Global Log-Rank Test
global_test = multivariate_logrank_test(
    df['duration_weeks'], df['primary_genre'], df['event_observed']
)
print(
    f'Omnibus Log-Rank: chi2 = {global_test.test_statistic:.2f}, p ='
    f' {global_test .p_value:.2e}\n'
)

# 2. Pairwise Log-Rank Tests
genres = df['primary_genre'].value_counts().index.tolist()
n_tests = len(genres)
summary_rows = []

for g in genres:
  in_g = df[df['primary_genre'] == g]
  out_g = df[df['primary_genre'] != g]

  kmf_in = KaplanMeierFitter().fit(in_g['duration_weeks'], in_g['event_observed'])
  lr = logrank_test(
      in_g['duration_weeks'],
      out_g['duration_weeks'],
      event_observed_A=in_g['event_observed'],
      event_observed_B=out_g['event_observed'],
  )

  summary_rows.append({
      'Genre': g,
      'N': len(in_g),
      'KM_Median': kmf_in.median_survival_time_,
      'Mean_Weeks': round(in_g['duration_weeks'].mean(), 2),
      'Chi2': round(lr.test_statistic, 2),
      'p_raw': lr.p_value,
      'p_bonferroni': min(lr.p_value * n_tests, 1.0),
      'Effect': (
          'Longer'
          if in_g['duration_weeks'].mean() > out_g['duration_weeks'].mean()
          else 'Shorter'
      ),
  })

genre_results = pd.DataFrame(summary_rows).sort_values('p_raw')
print(genre_results.to_string(index=False))


# Plotting 5 Signinficant Genres From Pairwise Log-Rank Tests
major_genres = genre_results[(genre_results['N'] >= 40) & (genre_results['p_bonferroni'] < 0.05)]['Genre'].tolist()
print(major_genres)


fig, ax = plt.subplots(figsize=(11, 6))
for g in major_genres:
  sub = df[df['primary_genre'] == g]
  kmf = KaplanMeierFitter().fit(
      sub['duration_weeks'],
      sub['event_observed'],
      label=f'{g} (n={len(sub)})',
  )
  kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2.0)

ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.2)
ax.set_title(
    'Kaplan-Meier Survival Curves by Primary Genre (n ≥ 40)',
    fontsize=13,
    fontweight='bold',
)
ax.set_xlabel('Duration in Netflix Global Top 10 (Weeks)', fontsize=11)
ax.set_ylabel('Survival Probability S(t)', fontsize=11)
ax.set_xlim(0, 18)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
plt.close()