# Netflix Streaming Longevity Analysis

## Project Overview
This project analyzes the factors driving the chart longevity of TV shows on Netflix. By combining historical weekly global chart data from Netflix with rich metadata from The Movie Database (TMDB), we apply survival analysis and statistical modeling to quantify how content characteristics (such as genre, episode count, and season count) and audience metrics influence how long a show remains in the Global Top 10.

## Project Goals
* **Model Retention:** Apply survival analysis (Kaplan-Meier estimates, Cox Proportional Hazards) to evaluate show decay off weekly charts.
* **Identify Key Predictors:** Measure the impact of structural attributes (genre, runtime, episode count) and audience feedback (ratings, vote volume) on chart duration.
* **Data Integration:** Build an automated pipeline merging weekly streaming records with external API metadata for exploratory and inferential modeling.

---

## File Explanations

### Current Python Scripts
* **`fetch_tmdb_metadata.py`**  
  Downloads the official Netflix historical Top 10 dataset and queries the TMDB API to retrieve metadata (genres, episode/season counts, ratings, and primary language) for all unique TV titles.
* **`build_master_dataset.py`**  
  Aggregates weekly Netflix records to compute each title's maximum chart longevity ($Y$), merges it with TMDB features ($X$), extracts primary genres, and exports `master_netflix_analysis.csv`.
* **`eda_presentation1.py`**  
  Computes summary statistics ($N$, mean, standard deviation) and exports exploratory data visualizations to the `plots/` directory for Presentation 1 slides.

### Directories & Data
* **`data/`**  
  * `all-weeks-global.tsv`: Raw weekly performance dataset directly from Netflix.
  * `netflix_tmdb_metadata.csv`: Extracted metadata features from the TMDB API.
  * `master_netflix_analysis.csv`: Merged master dataset used for statistical modeling.
* **`plots/`**  
  Will store plots that we create for our project. 

---

## Execution Order
To reproduce the data and merging from scratch, run the scripts in the following order:

```bash
python3 fetch_tmdb_metadata.py
python3 build_master_dataset.py
python3 eda_presentation1.py