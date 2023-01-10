import pandas as pd
import numpy as np

matrix_norm = pd.read_csv("pivot_table_normalized.csv", index_col=[0])
films_names = list(matrix_norm.columns)
user_vector_dictionary = {'Shrek (2001)': 5.0, 'Men in Black (a.k.a. MIB) (1997)': 5.0,
                          'Shawshank Redemption, The (1994)': 5.0, 'Pulp Fiction (1994)': 5.0,
                          'Matrix, The (1999)': 4.5, 'Terminator 2: Judgment Day (1991)': 4.0,
                          'Fight Club (1999)': 5.0, 'Toy Story (1995)': 5.0, 'Mask, The (1994)': 4.0}
user_id = matrix_norm.index.max() + 1

user_vector = pd.DataFrame(np.nan, index=[0], columns=matrix_norm.columns)
user_vector = user_vector.transpose().index.map(user_vector_dictionary)
user_vector = pd.DataFrame(user_vector).transpose()
user_vector.index = [user_id]
user_vector.columns = matrix_norm.columns
user_vector_norm = user_vector.subtract(user_vector.mean(axis=1), axis='rows')

matrix_norm = pd.concat([matrix_norm, user_vector_norm])
matrix_norm.to_csv('pivot_table_normalized.csv')

user_similarity = matrix_norm.T.corr()
user_similarity.drop(index=user_id, inplace=True)

n = 10
user_similarity_threshold = 0.3
similar_users = user_similarity[user_similarity[user_id] > user_similarity_threshold][user_id].sort_values(
    ascending=False)[:n]
user_id_watched = matrix_norm[matrix_norm.index == user_id].dropna(axis=1, how='all')
similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
similar_user_movies.drop(user_id_watched.columns, axis=1, inplace=True, errors='ignore')

item_score = {}
for i in similar_user_movies.columns:
    movie_ratings = similar_user_movies[i]
    total = 0
    count = 0
    for u in similar_users.index:
        if not pd.isna(movie_ratings[u]):
            score = similar_users[u] * movie_ratings[u]
            total += score
            count += 1
    item_score[i] = total / count

item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)[:20]
final = list(ranked_item_score['movie'])
print(*final, sep='\n')
