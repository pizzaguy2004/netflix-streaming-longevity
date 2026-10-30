import pandas as pd
import matplotlib.pyplot as plt

master = pd.read_csv("master_netflix_analysis.csv")

stats = master.describe()
summary = master.info()

fig, ax = plt.subplots(figsize=(12, 4)) # Adjust size to fit your data columns
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=stats.values, 
                 rowLabels=stats.index, 
                 colLabels=stats.columns, 
                 cellLoc='center', 
                 loc='center')

plt.savefig('dataset_summary_table.png', bbox_inches='tight', dpi=300)
print("📊 Successfully saved image as 'dataset_summary_table.png' in your folder!")

