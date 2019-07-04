import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from loading_data import load_movies, load_ratings, load_movies_transformed
from data_preprocessing import get_bag_of_words

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)

#movies, movies_credits, movies_keywords = load_movies()
movies = load_movies_transformed()
ratings = load_ratings()


def train_vectorizer():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(movies['bag_of_words'])
    return vectorizer


# returns a string containing words describing movies user likes
# and data frame of movies without those user has rated
def create_user_profile(user_id):
    users_movies = ratings.loc[ratings['userId'] == user_id]
    bag_of_words = ""

    # for removing movies that user rated
    movies_without_rated = movies

    for key, row in users_movies.iterrows():
        # remove a movie from the list
        movies_without_rated = movies_without_rated.drop(movies.index[movies['id'] == row['movieId']])

        if not float(row['rating']) >= 3.5:
            continue

        movie = movies.loc[movies['id'] == row['movieId']]
        if movie.empty:
            continue
        bag_of_words = bag_of_words + movie.iloc[0]['bag_of_words']

    set_of_words = set(bag_of_words.split())

    return " ".join(set_of_words), movies_without_rated


# returns list of indexes of recommended movies
def get_index_of_movies(similarity_matrix):
    ret_value = []
    similarity_list = similarity_matrix[0].tolist()
    for i in range(10):
        idx = similarity_list.index(max(similarity_list))
        ret_value.append(idx)
        similarity_list.pop(idx)
    return ret_value


# calculates similarity between a user and a movie
def jaccard_similarity(user, movie):
    str1 = set(user.split())
    str2 = set(movie.split())
    union = len(str1 | str2)
    intersection = len(str1 & str2)
    if intersection == 0 or union == 0:
        return 0
    return intersection/union


# returns a list of similarities between user and movies
def jaccard_similarity_matrix(user, movies):
    ret_value = []
    for key, row in movies.iterrows():
        ret_value.append(jaccard_similarity(user, row['bag_of_words']))
    return [ret_value]


def content_based(vectorizer, user_id):
    # user preferences vector
    user_profile, movies_to_watch = create_user_profile(user_id)

    # transforming data into numerical values using tfidf vectorizer
    movies_transformed = vectorizer.transform(movies_to_watch['bag_of_words'])
    user_profile_transformed = vectorizer.transform([user_profile])

    # calculate similarity between user and movies
    similarity_matrix = cosine_similarity(user_profile_transformed, movies_transformed)
    jac_similarity = jaccard_similarity_matrix(user_profile, movies_to_watch)

    # recommended
    index_list = get_index_of_movies(similarity_matrix)
    for i in index_list:
        print("_____________________")
        print(movies.iloc[i]['id'])
        print(movies.iloc[i]['title'])


def create_ratings_matrix(vectorizer):
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]
    matrix = np.zeros((n_users, n_movies))

    for row_idx in range(len(matrix)):
        # get user's rated movies
        user_rated_movies = get_user_rated_movies(row_idx+1)
        print ("____________________________")
        movies_transformed = vectorizer.transform(user_rated_movies['bag_of_words'])

        for col_idx in range(len(matrix[row_idx])):
            # get rating for a movie (col_idx) given by user (row_idx)
            rating = ratings.loc[(ratings['userId'] == row_idx+1) & (ratings['movieId'] == col_idx)]

            # if user didn't rate this movie, find most similar (by content) movie rated
            if rating.empty:
                movie = movies.loc[movies['id'] == col_idx].iloc[0]
                sim_matrix = cosine_similarity(vectorizer.transform([movie['bag_of_words']]), movies_transformed)
                sim_list = sim_matrix[0].tolist()
                idx = sim_list.index(max(sim_list))
                matrix[row_idx][col_idx] = user_rated_movies.get_value(idx, 'rating')
            else:
                matrix[row_idx][col_idx] = int(rating['rating'])
    return matrix


def create_ratings_matrix_clustering(clusters_of_movies):
    n_users = ratings.userId.unique().shape[0]
    n_movies = ratings.movieId.unique().shape[0]
    matrix = np.zeros((n_users, n_movies))

    for row_idx in range(len(matrix)):
        # get user's rated movies
        user_rated_movies = get_user_rated_movies(row_idx+1)

        cluster_dict = get_user_cluster_mean(user_rated_movies, clusters_of_movies)
        for col_idx in range(len(matrix[row_idx])):
            # get rating for a movie (col_idx) given by user (row_idx)
            rating = ratings.loc[(ratings['userId'] == row_idx+1) & (ratings['movieId'] == col_idx)]

            # if user didn't rate this movie, find most similar (by content) movie rated
            if rating.empty:
                value = cluster_dict[clusters_of_movies[col_idx]]
                matrix[row_idx][col_idx] = round(value[0] / value[1])
            else:
                matrix[row_idx][col_idx] = int(rating['rating'])
    return matrix


# returns data frame with 3 columns: bag_of_words of a movie and rating user gave it (and id of a movie)
def get_user_rated_movies(user_id):
    ratings_of_user = ratings.loc[ratings['userId'] == user_id]
    movies_rated = pd.DataFrame()
    idx = 0
    for key, row in ratings_of_user.iterrows():
        movie_id = int(row['movieId'])
        movie = movies.loc[movies['id'] == movie_id].iloc[0]
        movies_rated.set_value(idx, 'bag_of_words', movie['bag_of_words'])
        movies_rated.set_value(idx, 'rating', row['rating'])
        movies_rated.set_value(idx, 'id', row['movieId'])
        idx = idx + 1
    return movies_rated


# returns a dictionary - key: cluster label; value: [sum_of_ratings, num_of_ratings]
def get_user_cluster_mean(user_rated_movies, clusters):
    cluster_dict = {}
    for x in range(30):
        cluster_dict[x] = [0, 0]
    for key, row in user_rated_movies.iterrows():
        value = cluster_dict[clusters[int(row['id'])]]
        value[0] = value[0] + float(row['rating'])
        value[1] = value[1] + 1
        cluster_dict[clusters[int(row['id'])]] = value
    for x in cluster_dict.keys():
        if cluster_dict[x][0] == 0:
            value = cluster_dict[x]
            value[0] = 2.5
            value[1] = 1
    return cluster_dict


def get_collaborative_predictions(ratings_matrix, type, user_id):

    if type == "user":
        sim = cosine_similarity(ratings_matrix, ratings_matrix)
        calculated_similarities = sim.dot(ratings_matrix)
    else:
        sim = cosine_similarity(ratings_matrix.T, ratings_matrix.T)
        calculated_similarities = ratings_matrix.dot(sim)

    # dec because each user is "similar" to himself
    sum_similarities= list([np.abs(sim).sum(axis=1)])

    user_similarities = calculated_similarities[user_id]
    for i in range(len(user_similarities)):
        user_similarities[i] = user_similarities[i]/(sum_similarities[user_id]-1)

    rated_movies = get_user_rated_movies(user_id)

    predictions = pd.Series(user_similarities).sort_values(ascending=False)

    num = 0
    for i in range(len(predictions)):
        if i in rated_movies['id'].values.tolist():
            continue

        print("_____________________")
        print(movies.iloc[i]['id'])
        print(movies.iloc[i]['title'])
        num = num + 1
        if num == 10:
            break


if __name__ == '__main__':
    tfidf_vectorizer = train_vectorizer()
    clustering = KMeans(n_clusters=30)
    movies_transformed = tfidf_vectorizer.transform(movies['bag_of_words'])
    clustering.fit(movies_transformed)

    user_id = 430
    #content_based(tfidf_vectorizer, user_id)

    # finds most similar movie
    #ratings_matrix = create_ratings_matrix(tfidf_vectorizer)

    # uses clusters to calculate missing values
    ratings_matrix = create_ratings_matrix_clustering(clustering.predict(movies_transformed))

    get_collaborative_predictions(ratings_matrix, 'user', user_id)
