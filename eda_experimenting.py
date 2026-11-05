import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
wks_g = pd.read_csv('all-weeks-global.tsv', sep='\t')
mast_ntflx = pd.read_csv('master_netflix_analysis.csv')
ntflx_tmbd = pd.read_csv('netflix_tmdb_metadata.csv')

# Prepare data
plot_data = mast_ntflx.copy()

# Convert numerical variables
plot_data['vote_average'] = pd.to_numeric(plot_data['vote_average'], errors='coerce')
plot_data['number_of_episodes'] = pd.to_numeric(plot_data['number_of_episodes'], errors='coerce')
plot_data['duration_weeks'] = pd.to_numeric(plot_data['duration_weeks'], errors='coerce')

# Map language codes to full names
language_names = {
    'en': 'English',
    'ko': 'Korean',
    'es': 'Spanish',
    'ja': 'Japanese',
}

# Create language column
plot_data['language'] = (
    plot_data['original_language']
    .map(language_names)
    .fillna('Other')
)

# Set language order
language_order = [
    'English',
    'Korean',
    'Spanish',
    'Japanese',
    'Other'
]

# Set visual style
sns.set_theme(style='whitegrid')

# Assign consistent colors to each language
colors = sns.color_palette('colorblind', n_colors=5)

language_colors = dict(zip(language_order, colors))

# Common formatting function
def format_plot(title, xlabel, ylabel):
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.xticks(rotation=45, fontweight='bold')
    plt.tight_layout()
    plt.show()


# -----------------------------------
# Plot 1: Language vs. Vote Average
# -----------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=plot_data,
    x='language',
    y='vote_average',
    order=language_order,
    hue='language',
    palette=language_colors,
    legend=False
)

format_plot(
    'Rating Distribution by Original Language',
    'Original Language',
    'Average Rating'
)


# -----------------------------------
# Plot 2: Language vs. Episode Count
# -----------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=plot_data,
    x='language',
    y='number_of_episodes',
    order=language_order,
    hue='language',
    palette=language_colors,
    legend=False
)

format_plot(
    'Number of Episodes by Original Language',
    'Original Language',
    'Number of Episodes'
)


# -----------------------------------
# Plot 3: Language vs. Top 10 Duration
# -----------------------------------

plt.figure(figsize=(5.5, 6))

sns.boxplot(
    data=plot_data,
    x='language',
    y='duration_weeks',
    order=language_order,
    hue='language',
    palette=language_colors,
    legend=False
)

format_plot(
    'Netflix Global Top 10 Duration by Original Language',
    'Original Language',
    'Weeks in Global Top 10'
)


# -----------------------------------
# Display observations per language
# -----------------------------------

print("\nNumber of Shows per Language:")
print(plot_data['language'].value_counts().reindex(language_order))


# -----------------------------------
# Plot 4: Number of Shows by Language
# -----------------------------------

language_counts = plot_data['language'].value_counts().reindex(language_order)

plt.figure(figsize=(12, 6))

plt.bar(
    language_counts.index,
    language_counts.values,
    color=colors
)

plt.title(
    'Number of Netflix Shows by Original Language',
    fontweight='bold'
)
plt.xlabel('Original Language', fontweight='bold')
plt.ylabel('Number of Shows', fontweight='bold')
plt.xticks(rotation=45, fontweight='bold')

# Add count labels above bars
for i, count in enumerate(language_counts.values):
    plt.text(
        i,
        count + 2,
        str(count),
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()