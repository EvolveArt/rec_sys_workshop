from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, SQLContext

from funk_svd import SVD

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


class SVDRecommender:
    def __init__(
        self,
        train_df,
        regParam,
        n_factors,
        min_rating=0,
        max_rating=100,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="ratings",
    ):
        self.userCol = userCol
        self.itemCol = itemCol
        self.ratingCol = ratingCol
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.train_df = train_df
        self.regParam = regParam
        self.rank = n_factors

    def train__SGD(self, learning_rate=0.001, n_epochs=100):
        """
        Train the MF model with SGD optimizer using fast funkSVD library
        """

        # Rename the user,item,rating cols to fit the library requirements
        train_df_sgd = self.train_df.rename(
            columns={
                self.userCol: "u_id",
                self.itemCol: "i_id",
                self.ratingCol: "rating",
            }
        )

        # Create a validation set with 10% of training set
        val_df = train_df_sgd.sample(frac=0.1, random_state=42)
        train_df_sgd.drop(val_df.index.tolist(), inplace=True)

        self.svd = SVD(
            learning_rate=learning_rate,
            regularization=self.regParam,
            n_epochs=n_epochs,
            n_factors=self.rank,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
        )
        self.svd.fit(X=train_df_sgd, X_val=val_df, early_stopping=False, shuffle=True)

    def evaluate__SGD(self, test_df):
        """
        Evaluate the funkSVD model
        metrics: RMSE, MAE
        """

        # Rename the user,item,rating cols to fit the library requirements
        test_df_sgd = test_df.rename(
            columns={
                self.userCol: "u_id",
                self.itemCol: "i_id",
                self.ratingCol: "rating",
            }
        )

        predictions = self.svd.predict(test_df_sgd[["u_id", "i_id"]])
        y_true = test_df[self.ratingCol].fillna(53)

        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))

        print(
            f"Metrics on test set for funkSVD model : \n RMSE={rmse:0.2f} \n MAE={mae:0.2f}"
        )

    def recommend__SGD(self, user, topn=10):
        # Create a dataframe with two columns [userIDs, itemIDs]
        allItems = self.train_df[self.itemCol].unique().tolist()
        columns = ["u_id", "i_id"]
        user_df = pd.DataFrame(
            {columns[0]: [user for _ in range(len(allItems))], columns[1]: allItems}
        )

        # Compute predicted ratings and sort them
        ratings = self.svd.predict(user_df)

        recs = {}
        for _ in range(topn):
            max_index = np.argmax(ratings)
            max_rating = ratings.pop(max_index)
            recs[allItems[max_index]] = max_rating

        return pd.DataFrame.from_dict(recs, orient="index")

