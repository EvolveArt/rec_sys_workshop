import time

from functools import wraps
from math import trunc
import scipy.sparse as sp
import numpy as np


def normalize(x):
    """
    param x: vecteur a normaliser
    return: x normalisé sur [0, 1]
    """
    x = x.astype(float)
    x_sum = x.sum()
    x_num = x.astype(bool).sum()
    x_mean = x_sum / x_num
    if x.std() == 0:
        return 0.0
    return (x - x_mean) / (x.max() - x.min())


def get_items_rated_by(user_id, ratings_matrix):
    """
    param user_id: id de l'utilisateur recherché
    param ratings_matrix: matrice d'interactions
    return: items notés par l'utilisateur user_id
    """
    user_ratings = ratings_matrix.loc[[user_id]]
    items = user_ratings[user_ratings.columns[~user_ratings.isnull().all()]]

    return items.columns.values.tolist()


def get_users_who_rated(item_id, ratings_matrix):
    """
    param item_id: id de l'item recherché
    param ratings_matrix: matrice d'interactions
    return: utilisateurs ayant notés l'item d'id item_id
    """
    ratings_matrix_T = ratings_matrix.copy().transpose()
    item_ratings = ratings_matrix_T.loc[[item_id]]
    users = item_ratings[item_ratings.columns[~item_ratings.isnull().all()]]

    return users.columns.values.tolist()


def timer(text=""):
    """Decorator, prints execution time of the function decorated.
    Args:
        text (string): text to print before time display.
    Examples:
        >>> @timer(text="Greetings took ")
        ... def say_hi():
        ...    time.sleep(1)
        ...    print("Hey! What's up!")
        ...
        >>> say_hi()
        Hey! What's up!
        Greetings took 1 sec
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            hours = trunc((end - start) / 3600)
            minutes = trunc((end - start) / 60)
            seconds = round((end - start) % 60)

            if hours > 1:
                print(
                    text + "{} hours {} min and {} sec".format(hours, minutes, seconds)
                )
            elif hours == 1:
                print(
                    text + "{} hour {} min and {} sec".format(hours, minutes, seconds)
                )
            elif minutes >= 1:
                print(text + "{} min and {} sec".format(minutes, seconds))
            else:
                print(text + "{} sec".format(seconds))

            return result

        return wrapper

    return decorator


def build_sparse_matrix(df, labels):
    """
    Build the sparse matrix associated to a pandas dataframe of 3 columns
    param df: dataframe to convert to sparse matrix
    param labels: column labels eg. ['user_id', 'item_id', 'ratings']
    return: scipy sparse matrix
    """

    userCol, itemCol, ratingCol = labels

    new_df = df.copy()

    new_df[userCol] = new_df[userCol].astype("category").cat.codes
    new_df[itemCol] = new_df[itemCol].astype("category").cat.codes

    users = list(np.sort(new_df[userCol].unique()))
    items = list(np.sort(new_df[itemCol].unique()))
    ratings = list(new_df[ratingCol])

    rows = new_df[userCol].astype(float)
    cols = new_df[itemCol].astype(float)

    data_sparse = sp.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(items)))

    sparsity = 1 - data_sparse.nnz / (data_sparse.shape[0] * data_sparse.shape[1])
    print(f"Data Sparsity : {sparsity*100:0.5f} %")

    return data_sparse


# def build_sim_matrix():

#         self.train_df = self.train_df.astype({"ratings": float})
#         self.train_df["avg"] = self.train_df.groupby(self.user_id)["ratings"].transform(
#             lambda x: normalize(x)
#         )

#         train_df_na = self.train_df.fillna(0)

#         train_df_na = train_df_na.astype({self.user_id: "category"})
#         train_df_na = train_df_na.astype({self.item_id: "category"})

#         coo = coo_matrix(
#             (
#                 train_df_na["avg"].astype(float),
#                 (
#                     train_df_na[self.item_id].cat.codes.copy(),
#                     train_df_na[self.user_id].cat.codes.copy(),
#                 ),
#             )
#         )

#         overlap_mat = (
#             coo.astype(bool).astype(int).dot(coo.transpose().astype(bool).astype(int))
#         )

#         cor = cosine_similarity(coo, dense_output=False)
#         cor = cor.multiply(cor > self.min_sim)
#         cor = cor.multiply(overlap_mat > self.min_overlap)

#         items = dict(enumerate(train_df_na[self.item_id].cat.categories))

#         return cor, items
