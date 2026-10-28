import os
import pandas as pd
import requests
import time

# downloads netflix datasets locally
NETFLIX_TSV = "all-weeks-global.tsv"
NETFLIX_URL = "https://top10.netflix.com/data/all-weeks-global.tsv"

if not os.path.exists(NETFLIX_TSV):
    print("Downloading Netflix Top 10 dataset...")
    response = requests.get(NETFLIX_URL, stream=True)
    response.raise_for_status()

    with open(NETFLIX_TSV, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")
else:
    print("Local Netflix dataset found.")

# load netflix data
netflix_df = pd.read_csv(NETFLIX_TSV, sep="\t")
print(f"Successfully loaded {len(netflix_df)} rows from Netflix dataset.")

# TMDB api key
API_KEY = "36455b183a91eb46bcc03c412113d2f1"  # Keep your actual API key here


def get_tv_details(tmdb_id, api_key):
    details_url = f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={api_key}"
    res = requests.get(details_url).json()

    return {
        "number_of_episodes": res.get("number_of_episodes"),
        "number_of_seasons": res.get("number_of_seasons"),
        "episode_run_time": (
            res.get("episode_run_time", [None])[0]
            if res.get("episode_run_time")
            else None
        ),
        "genres": [g["name"] for g in res.get("genres", [])],
        "status": res.get("status"),
        "in_production": res.get("in_production"),
    }


# --- 4. FETCH TMDB METADATA FOR UNIQUE SHOWS ---
unique_shows = netflix_df[netflix_df["category"].str.contains("TV")][
    "show_title"
].unique()
print(f"Total unique TV shows in dataset: {len(unique_shows)}")

metadata_list = []

# Fetch first 50 titles as a test run
for title in unique_shows:
    search_url = f"https://api.themoviedb.org/3/search/tv?api_key={API_KEY}&query={title}"
    res = requests.get(search_url).json()

    if res.get("results"):
        top_match = res["results"][0]
        show_id = top_match["id"]

        details = get_tv_details(show_id, API_KEY)

        metadata_list.append(
            {
                "show_title": title,
                "tmdb_id": show_id,
                "original_language": top_match.get("original_language"),
                "vote_average": top_match.get("vote_average"),
                "vote_count": top_match.get("vote_count"),
                "first_air_date": top_match.get("first_air_date"),
                "genres": ", ".join(details["genres"]),
                "number_of_episodes": details["number_of_episodes"],
                "number_of_seasons": details["number_of_seasons"],
            }
        )

    time.sleep(0.05)  # Rate limiting

# --- 5. SAVE MERGED PREDICTOR DATASET ---
tmdb_df = pd.DataFrame(metadata_list)
tmdb_df.to_csv("netflix_tmdb_metadata.csv", index=False)
print("Saved enriched metadata to netflix_tmdb_metadata.csv!")
print(tmdb_df.head())