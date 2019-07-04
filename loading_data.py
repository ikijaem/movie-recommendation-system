import pandas as pd
from data_preprocessing import get_bag_of_words


def load_movies_transformed():
    movies_transformed = pd.read_csv("data/movies_metadata_transformed.csv", encoding="utf-8")
    return movies_transformed


def load_movies():
    # movies
    #cols = ["adult", "genres", "id", "original_title", "spoken_languages", "title", "production_companies"]
    df_movies = pd.read_csv("data/movies_metadata.csv", encoding="utf-8")
    # credits
    df_credits = pd.read_csv("data/credits.csv")
    # keywords
    df_keywords = pd.read_csv("data/keywords.csv")
    # creates new column "bag_of_words" (contains info from movies_metadata, keywords and credits)
    for key, row in df_movies.iterrows():
        # getting info about current movie from other data sets
        credits_of_movie = df_credits.loc[df_credits['id'] == int(row['id'])]
        keywords_of_movie = df_keywords.loc[df_keywords['id'] == int(row['id'])]

        # some movies don't exist in keywords/credits data sets
        if credits_of_movie.empty:
            credits_of_movie = {'cast': '[]', 'crew': '[]'}
        else:
            credits_of_movie = credits_of_movie.iloc[0]
        if keywords_of_movie.empty:
            keywords_of_movie = {'keywords': '[]'}
        else:
            keywords_of_movie = keywords_of_movie.iloc[0]
        bag_of_words = get_bag_of_words(row, credits_of_movie, keywords_of_movie)
        if bag_of_words == "":
            bag_of_words = "[]"
        df_movies.set_value(key, 'bag_of_words', bag_of_words)

    # removing transformed columns from data set
    #cols_to_remove = ["genres", "spoken_languages", "production_companies", "adult"]
    #df_movies = df_movies.drop(cols_to_remove, axis=1)
    return df_movies, df_credits, df_keywords


def load_ratings():
    ratings = pd.read_csv("data/ratings_mini.csv")
    return ratings
