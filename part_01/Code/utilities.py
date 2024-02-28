import os
import time

import requests
import pandas as pd
import numpy as np 


def get_data_block_of_years(years: tuple[int, int], country: str, data_directory: str, api_key: str, df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    Iterates through a list of tuples containing a year, country name and direction of change in freedom score.
    Fetches headlines about that country from the NYT API for that year and the year following.
    Saves the data to a parquet file, as a new file if none exists or appending to the one already there
    Args:
        years ([tuple[int, int]]): beginning and end of time period for which to fetch data
        country: country name
        data_directory: destination directory where the parquet file should be saved
        api_key: user's NYT API key
        df: optional; DataFrame containing NYT headline data from some previous requests
    Returns:
        DataFrame with headline data for the given country & year(s)
    """

    file_name = f"{data_directory}/{country.lower()}.parquet"
    for year in range(years[0], years[-1]+1):
        for month in range(1,13):
            print(f"getting data for {country} in month {month}, year {year}")
            df = get_data(country, month, year, df, api_key)
            time.sleep(10)
        try:
            # Try to read the existing data
            data = pd.read_parquet(file_name)
            # Append the new data
            df = pd.concat([data, df])
            df = df.drop_duplicates(df)
        except:
            # If the file doesn't exist, use new_data as df
            print('first time through')
        df.to_parquet(file_name, engine='pyarrow')
        print(f"saved data to {file_name}")
    return df


def get_data_by_year(change_years: list[tuple[int, str, str]], data_directory: str, api_key: str, df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    Iterates through a list of tuples containing a year, country name and direction of change in freedom score.
    Fetches headlines about that country from the NYT API for that year and the year following.
    Saves the data to a parquet file, as a new file if none exists or appending to the one already there
    Args:
        change_years (list[tuple[int, str, str]]): return value of the `score_change_years` functions
        data_directory: destination directory where the parquet file should be saved
        api_key: user's NYT API key
        df: optional; DataFrame containing NYT headline data from some previous requests
    Returns:
        DataFrame with headline data for the given country & year(s)
    """
    countries = [ changes[2] for changes in change_years]
    if len(set(countries)) > 1:
        print(f"Expected one country in change_years but found {set(countries)}.")
        return
    
    years_checked = []
    for info in change_years:
        country = info[2]
        label = info[1]
        years = (info[0], info[0] + 1)
        file_name = f"{data_directory}/{country.lower()}.parquet"
        print(file_name)
        for year in years:
            if year in years_checked:
                print(f"Already fetched data for {year}")
            else:
                for month in range(1,13):
                    print(f"getting data for {country} in month {month}, year {year}")
                    df = get_data(country, month, year, df, api_key)
                    df["label"] = label
                    time.sleep(10)
                try:
                    # Try to read the existing data
                    data = pd.read_parquet(file_name)
                    # Append the new data
                    df = pd.concat([data, df])
                    df = df.drop_duplicates(df)
                except:
                    # If the file doesn't exist, use new_data as df
                    print('first time through')
                df.to_parquet(file_name, engine='pyarrow')
                print(f"saved data to {file_name}")
                years_checked.append(year)
    return df
    

def countries_change(df: pd.DataFrame, threshold: int = 20, direction: str = "down") -> list[str]:
    """
    Args:
        df: DataFrame from FOTN 2011-2023 Score Data.xlsx, after some column-name fixes
        threshold: Magnitude of change in a country's "total_score" to be included. Defaults to 20.
        direction: Direction of change in a country's "total_score". Defaults to "down".
    Returns:
        list of countries whose "total_score" has changed by at least the threshold amount in the direction given
    """
    country_score = df.groupby('country')
    sign_maxmin=np.sign(country_score['total_score'].apply(np.argmax)-country_score['total_score'].apply(np.argmin))
    if direction == "down":
        mask_score=((country_score.max()['total_score']-country_score.min()['total_score']) >= threshold) & (sign_maxmin>0)
    else:
        mask_score=((country_score.max()['total_score']-country_score.min()['total_score']) >= threshold) & (sign_maxmin<0)

    return [c for c in set(df['country']) if (mask_score[c])]


def score_change_years(df: pd.DataFrame, country: str, threshold: int = 5) -> list[tuple[int, str, str]]:
    """
    Args:
        df: DataFrame from FOTN 2011-2023 Score Data.xlsx, after some column-name fixes
        country: name of country to check
        threshold: Magnitude of change in a country's "total_score" to be included. Defaults to 5.
        
    Returns:
        list of tuples where the first value is the year BEFORE the change, the second value is the direction
        of the change (increasing or decreasing) & the third is the name of the country
    """
    
    df =  df.loc[df["country"] == country]
    scores = list(df["total_score"])
    years = list(df["year"])
        
    changes = []
    for index in range(len(scores)-1):
        if abs(scores[index] - scores[index+1]) >= threshold:
            year = years[index]
            direction = "increasing" if scores[index] - scores[index+1] > 0 else "decreasing"
            changes.append((year, direction, country))

    return changes


def combine_parquet_files(folder: str) -> pd.DataFrame:
    files = os.listdir(folder)
    dfs = []
    for file in files:
        file_path = f"{folder}/{file}"
        with open(file_path, "rb") as f:
            dfs.append(pd.read_parquet(f))
    return pd.concat(dfs)
   
# **************************************************************************
# Helper functions: used in the functions above, not called directly   

def get_data(keyword: str, month: int, year: int, data_df: pd.DataFrame, api_key: str) -> pd.DataFrame | None:
    nyt_url = "https://api.nytimes.com/svc/archive/v1"
    full_url = f"{nyt_url}/{year}/{month}.json?api-key={api_key}"
    response = requests.get(full_url)
    data = response.json()
    
    if response.status_code == 200:
        articles = []
        for story in data["response"]["docs"]:
            for keyword_dict in story["keywords"]:
                if keyword in keyword_dict.values():
                    article = {
                        "abstract": story["abstract"],
                        "headline": story["headline"]["main"],
                        "pub_date": story["pub_date"],
                        "year": year,
                        "section_name": story["section_name"],
                        "news_desk": story["news_desk"],
                        "keyword": keyword
                    }
                    articles.append(article)
        article_df = pd.DataFrame.from_dict(articles)
        return pd.concat([data_df, article_df], ignore_index=True)        
        
    if response.status_code == 429:
        print("Got 'too many requests' error. Going to sleep (zzzzzzzz)")
        time.sleep(10)
        print("Retrying now")
        get_data(keyword, month, year, data_df, api_key)
    else: 
        print(f"Something went wrong! Request status call = {response.status_code}")
        print(f"Here's the url we tried: {full_url}")
        return