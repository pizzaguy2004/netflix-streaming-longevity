from pathlib import Path
import pandas as pd
import requests
import time

# downloads netflix datasets locally
DATA_DIR = Path(__file__).resolve().parent / "data"
NETFLIX_TSV = DATA_DIR / "all-weeks-global.tsv"
NETFLIX_URL = "https://top10.netflix.com/data/all-weeks-global.tsv"
SERIES_OUTPUT = DATA_DIR / "netflix_tmdb_metadata.csv"
SEASON_OUTPUT = DATA_DIR / "netflix_tmdb_season_metadata.csv"

if not NETFLIX_TSV.exists():
    print("Downloading Netflix Top 10 dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    response = requests.get(details_url, timeout=30)
    response.raise_for_status()
    res = response.json()

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
        "season_numbers": [
            season.get("season_number")
            for season in res.get("seasons", [])
            if season.get("season_number", 0) > 0
        ],
    }


def get_season_details(tmdb_id, season_number, api_key):
    season_url = (
        f"https://api.themoviedb.org/3/tv/{tmdb_id}/season/"
        f"{season_number}?api_key={api_key}"
    )
    response = requests.get(season_url, timeout=30)
    response.raise_for_status()
    return response.json()


# --- 4. FETCH TMDB METADATA FOR UNIQUE SHOWS ---
unique_shows = netflix_df[netflix_df["category"].str.contains("TV")][
    "show_title"
].unique()
print(f"Total unique TV shows in dataset: {len(unique_shows)}")

metadata_list = []
season_metadata_list = []

# Fetch show metadata and season-level metadata for each matched TV title.
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

        for season_number in details["season_numbers"]:
            try:
                season = get_season_details(show_id, season_number, API_KEY)
                episodes = season.get("episodes", [])
                runtimes = [
                    episode["runtime"]
                    for episode in episodes
                    if episode.get("runtime") is not None
                ]
                season_metadata_list.append(
                    {
                        "show_title": title,
                        "tmdb_id": show_id,
                        "season_number": season_number,
                        "season_name": season.get("name"),
                        "season_air_date": season.get("air_date"),
                        "season_episode_count": len(episodes),
                        "season_vote_average": season.get("vote_average"),
                        "season_vote_count": season.get("vote_count"),
                        "avg_episode_runtime": (
                            sum(runtimes) / len(runtimes) if runtimes else None
                        ),
                    }
                )
            except requests.RequestException as exc:
                print(
                    f"Could not fetch season {season_number} for {title}: {exc}"
                )
            time.sleep(0.05)

    time.sleep(0.05)  # Rate limiting

# saves merged data
tmdb_df = pd.DataFrame(metadata_list)
tmdb_df.to_csv(SERIES_OUTPUT, index=False)
season_df = pd.DataFrame(season_metadata_list)
season_df.to_csv(SEASON_OUTPUT, index=False)
print(f"Saved show metadata to {SERIES_OUTPUT}!")
print(f"Saved season metadata to {SEASON_OUTPUT}!")
print(tmdb_df.head())
