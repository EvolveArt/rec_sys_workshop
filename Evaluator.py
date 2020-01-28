import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, recsys, test_df, threshold=0.5):
        self.recsys = recsys
        self.test_df = test_df

    def binary_eval(self):
        # predict all the ratings for test data
        ratings_test = self.test_df.assign(
            predicted_rating=pd.Series(np.zeros(self.test_df.shape[0]))
        )
        for index, row_rating in ratings_test.iterrows():
            predicted_rating = self.recsys.predict(
                row_rating[self.recsys.user_col], row_rating[self.recsys.item_col]
            )
            ratings_test.loc[index, "predicted_rating"] = predicted_rating

        tp = ratings_test.query(
            "(ratings >= @threshold) & (predicted_rating >= @threshold)"
        ).shape[0]
        fp = ratings_test.query(
            "(ratings < @threshold) & (predicted_rating >= @threshold)"
        ).shape[0]
        fn = ratings_test.query(
            "(ratings >= @threshold) & (predicted_rating < @threshold)"
        ).shape[0]

        # calculate the precision and recall
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (precision, recall)
