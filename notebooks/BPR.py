import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp

from utils import build_sparse_matrix

from tqdm import tqdm

tf.disable_v2_behavior()


class BPRRecommender:
    def __init__(
        self, ratings_df, userCol="user_id", itemCol="item_id", ratingCol="ratings"
    ):
        self.userCol = userCol
        self.itemCol = itemCol
        self.ratingCol = ratingCol

        self.ratings_df = ratings_df
        self.data_sparse = build_sparse_matrix(
            ratings_df, [userCol, itemCol, ratingCol]
        )

        self.graph = tf.Graph()
        self.users = list(np.sort(ratings_df[userCol].unique()))
        self.items = list(np.sort(ratings_df[itemCol].unique()))

    @staticmethod
    def init_variable(size, dim, name=None):
        """
        Helper function to init a new variable with uniform random valuers
        """
        std = np.sqrt(2 / dim)
        return tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)

    @staticmethod
    def embed(inputs, size, dim, name=None):
        """
        Helper function to get a Tensorflow variable and create an embedding lookup 
        in order to map our user and item indices to vector
        """
        emb = BPRRecommender.init_variable(size, dim, name)
        return tf.nn.embedding_lookup(emb, inputs)

    def get_variable(self, session, name):
        """
        Helper function to get the value of Tensorflow variable by name
        """
        v = self.graph.get_operation_by_name(name)
        v = v.values()[0]
        v = v.eval(session=session)

        return v

    def set_hyperparameters(
        self,
        num_factors=64,
        lambda_user=1e-6,
        lambda_item=1e-6,
        lambda_bias=1e-6,
        learning_rate=0.005,
    ):
        self.num_factors = num_factors
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.lambda_bias = lambda_bias
        self.lr = learning_rate

    def build_graph(self):
        with self.graph.as_default():
            """
            Loss function: 
            -SUM ln σ(xui - xuj) + λ(w1)**2 + λ(w2)**2 + λ(w3)**2 ...
            ln = the natural log
            σ(xuij) = the sigmoid function of xuij.
            λ = lambda regularization value.
            ||W||**2 = the squared L2 norm of our model parameters.

            """

            # Input into our model, in this case our user (u),
            # known item (i) an unknown item (i) triplets.
            self.u = tf.placeholder(tf.int32, shape=(None, 1))
            self.i = tf.placeholder(tf.int32, shape=(None, 1))
            self.j = tf.placeholder(tf.int32, shape=(None, 1))

            # User feature embedding
            u_factors = self.embed(
                self.u, len(self.users), self.num_factors, "user_factors"
            )  # U matrix

            # Known and unknown item embeddings
            item_factors = self.init_variable(
                len(self.items), self.num_factors, "item_factors"
            )  # V matrix
            i_factors = tf.nn.embedding_lookup(item_factors, self.i)
            j_factors = tf.nn.embedding_lookup(item_factors, self.j)

            # i and j bias embeddings.
            item_bias = self.init_variable(len(self.items), 1, "item_bias")
            i_bias = tf.nn.embedding_lookup(item_bias, self.i)
            i_bias = tf.reshape(i_bias, [-1, 1])
            j_bias = tf.nn.embedding_lookup(item_bias, self.j)
            j_bias = tf.reshape(j_bias, [-1, 1])

            # Calculate the dot product + bias for known and unknown
            # item to get xui and xuj.
            xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
            xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)

            # We calculate xuij.
            xuij = xui - xuj

            # Calculate the mean AUC (area under curve).
            # if xuij is greater than 0, that means that
            # xui is greater than xuj (and thats what we want).
            self.u_auc = tf.reduce_mean(tf.to_float(xuij > 0))

            # Output the AUC value to tensorboard for monitoring.
            tf.summary.scalar("auc", self.u_auc)

            # Calculate the squared L2 norm ||W||**2 multiplied by λ.
            l2_norm = tf.add_n(
                [
                    self.lambda_user * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
                    self.lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
                    self.lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
                    self.lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
                    self.lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias)),
                ]
            )

            # Calculate the loss as ||W||**2 - ln σ(Xuij)
            # loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
            self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij))) + l2_norm

            # Train using the Adam optimizer to minimize
            # our loss function.
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.step = opt.minimize(self.loss)

            # Initialize all tensorflow variables.
            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

    def train(self, epochs, batches, samples, save_path="models/bpr-recsys-1.0"):

        uids, iids = self.data_sparse.nonzero()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        for epoch in range(epochs):
            for _ in range(batches):

                # We want to sample one known and one unknown
                # item for each user.
                import random

                idx = random.sample(range(len(uids)), samples)

                batch_u = [[uids[idxx]] for idxx in idx]
                batch_i = [[iids[idxx]] for idxx in idx]

                idx = random.sample(range(len(self.items)), samples)
                batch_j = [[idxx] for idxx in idx]

                # Feed our users, known and unknown items to
                # our tensorflow graph.
                feed_dict = {self.u: batch_u, self.i: batch_i, self.j: batch_j}

                # We run the session.
                _, l, auc = self.sess.run([self.step, self.loss, self.u_auc], feed_dict)

            print(" Epoch %d : Loss: %.3f | AUC: %.3f" % (epoch, l, auc))

            self.saver.save(self.sess, save_path)

    def find_similar_items(self, item_lookup, item=None, num_items=10):
        """Find items similar to an item.
        Args:
            item (str): The name of the item we want to find similar items for
            num_items (int): How many similar items we want to return.
            item_lookup: dataframe with a item_name/item_id columns, has to be item_name
        Returns:
            similar (pandas.DataFrame): DataFrame with num_items item names and scores
        """

        # Grab our User matrix U
        user_vecs = self.get_variable(self.sess, "user_factors")

        # Grab our Item matrix V
        item_vecs = self.get_variable(self.sess, "item_factors")

        # Grab our item bias
        item_bi = self.get_variable(self.sess, "item_bias").reshape(-1)

        # Get the item id
        item_id = int(item_lookup[item_lookup.item_name == item][self.itemCol])

        # Get the item vector for our item_id and transpose it.
        item_vec = item_vecs[item_id].T

        # Calculate the similarity between Lady GaGa and all other items
        # by multiplying the item vector with our item_matrix
        scores = np.add(item_vecs.dot(item_vec), item_bi).reshape(1, -1)[0]

        # Get the indices for the top 10 scores
        top_10 = np.argsort(scores)[::-1][:num_items]

        # We then use our lookup table to grab the names of these indices
        # and add it along with its score to a pandas dataframe.
        items, item_scores = [], []

        for idx in top_10:
            items.append(
                item_lookup.item_name.loc[item_lookup.artist_id == idx].iloc[0]
            )
            item_scores.append(scores[idx])

        similar = pd.DataFrame({"item": items, "score": item_scores})

        return similar

    def make_recommendation(self, item_lookup, user_id=None, num_items=10):
        """Recommend items for a given user given a trained model
        Args:
            user_id (int): The id of the user we want to create recommendations for.
            num_items (int): How many recommendations we want to return.
            item_lookup: dataframe with a item_name/item_id columns (match case)
        Returns:
            recommendations (pandas.DataFrame): DataFrame with num_items item names and scores
        """

        # Grab our user matrix U
        user_vecs = self.get_variable(self.sess, "user_factors")

        # Grab our item matrix V
        item_vecs = self.get_variable(self.sess, "item_factors")

        # Grab our item bias
        item_bi = self.get_variable(self.sess, "item_bias").reshape(-1)

        # Calculate the score for our user for all items.
        rec_vector = np.add(user_vecs[user_id, :].dot(item_vecs.T), item_bi)

        # Grab the indices of the top users
        item_idx = np.argsort(rec_vector)[::-1][:num_items]

        # Map the indices to item names and add to dataframe along with scores.
        items, scores = [], []

        for idx in item_idx:
            items.append(
                item_lookup.item_name.loc[item_lookup.artist_id == idx].iloc[0]
            )
            scores.append(rec_vector[idx])

        recommendations = pd.DataFrame({"item": items, "score": scores})

        return recommendations

    def close_session(self):
        self.sess.close()
