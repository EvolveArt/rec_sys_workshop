import pandas as pd
import numpy as np
from utils import normalize
import pickle


class ItemCFRecommender:
    def __init__(
        self, train_df, user_col="user_id", item_col="item_id", load_sim_matrix=True
    ):
        self.train_df = train_df
        self.load_sim_matrix = load_sim_matrix
        self.user_col = user_col
        self.item_col = item_col

    def build_sim_matrix(self, ratings):

        # adjust the ratings
        rating_mean = (
            ratings.groupby([self.user_col], as_index=False, sort=False)
            .mean()
            .rename(columns={"ratings": "rating_mean"})[[self.user_col, "rating_mean"]]
        )

        # print(rating_mean.head())

        adjusted_ratings = pd.merge(
            ratings, rating_mean, on=self.user_col, how="left", sort=False
        )
        adjusted_ratings["rating_adjusted"] = (
            adjusted_ratings["ratings"] - adjusted_ratings["rating_mean"]
        )
        # replace 0 adjusted rating values to 1*e-8 in order to avoid 0 denominator
        adjusted_ratings.loc[
            adjusted_ratings["rating_adjusted"] == 0, "rating_adjusted"
        ] = 1e-8

        # define sim matrix
        w_matrix_columns = ["item_1", "item_2", "similarity"]

        w_matrix = pd.DataFrame(columns=w_matrix_columns)

        # load sim matrix from pickle file
        if self.load_sim_matrix:
            with open("../models/w_matrix.pkl", "rb") as input:
                w_matrix = pickle.load(input)
            input.close()

        # calculate the similarity values
        else:
            distinct_items = np.unique(adjusted_ratings[self.item_col])

            i = 0
            # for each item_1 in all items
            for item_1 in distinct_items:

                if i % 10 == 0:
                    print(i, "out of ", len(distinct_items))

                # extract all users who rated item_1
                user_data = adjusted_ratings[adjusted_ratings[self.item_col] == item_1]
                distinct_users = np.unique(user_data[self.user_col])
                print(f"{len(distinct_users)} rated {item_1}")

                # record the ratings for users who rated both item_1 and item_2
                record_row_columns = [
                    self.user_col,
                    "item_1",
                    "item_2",
                    "rating_adjusted_1",
                    "rating_adjusted_2",
                ]
                record_item_1_2 = pd.DataFrame(columns=record_row_columns)
                # for each customer C who rated item_1
                for c_userid in distinct_users:
                    # print(
                    #     f"build weight matrix for customer {c_userid}, item_1 {item_1}"
                    # )
                    # the customer's rating for item_1
                    c_item_1_rating = user_data[user_data[self.user_col] == c_userid][
                        "rating_adjusted"
                    ].iloc[0]
                    # extract items rated by the customer excluding item_1
                    c_user_data = adjusted_ratings[
                        (adjusted_ratings[self.user_col] == c_userid)
                        & (adjusted_ratings[self.item_col] != item_1)
                    ]
                    c_distinct_items = np.unique(c_user_data[self.item_col])

                    # for each item rated by customer C as item_2
                    for item_2 in c_distinct_items:
                        # the customer's rating for item_2
                        c_item_2_rating = c_user_data[
                            c_user_data[self.item_col] == item_2
                        ]["rating_adjusted"].iloc[0]
                        record_row = pd.Series(
                            [
                                c_userid,
                                item_1,
                                item_2,
                                c_item_1_rating,
                                c_item_2_rating,
                            ],
                            index=record_row_columns,
                        )
                        record_item_1_2 = record_item_1_2.append(
                            record_row, ignore_index=True
                        )

                    # calculate the similarity values between item_1 and the above recorded items
                    distinct_item_2 = np.unique(record_item_1_2["item_2"])
                    # for each item 2
                    for item_2 in distinct_item_2:
                        # print(f"Computing Sim({item_1}, {item_2})..")
                        paired_item_1_2 = record_item_1_2[
                            record_item_1_2["item_2"] == item_2
                        ]
                        sim_value_numerator = (
                            paired_item_1_2["rating_adjusted_1"]
                            * paired_item_1_2["rating_adjusted_2"]
                        ).sum()
                        sim_value_denominator = np.sqrt(
                            np.square(paired_item_1_2["rating_adjusted_1"]).sum()
                        ) * np.sqrt(
                            np.square(paired_item_1_2["rating_adjusted_2"]).sum()
                        )

                        if sim_value_denominator == 0:
                            sim_value_denominator = 1e-8

                        sim_value = sim_value_numerator / sim_value_denominator
                        w_matrix = w_matrix.append(
                            pd.Series(
                                [item_1, item_2, sim_value], index=w_matrix_columns
                            ),
                            ignore_index=True,
                        )

                i = i + 1

            # output weight matrix to pickle file
            with open("../models/w_matrix.pkl", "wb") as output:
                pickle.dump(w_matrix, output, pickle.HIGHEST_PROTOCOL)
            output.close()

        return w_matrix, adjusted_ratings

    def predict(self, user, item, w_matrix, adjusted_ratings):

        # compute the user_mean
        user_other_ratings = adjusted_ratings[adjusted_ratings[self.user_col] == user]
        user_distinct_items = np.unique(user_other_ratings[self.item_col])

        # calculate P(u, i)

        sum_weighted_other_ratings = 0
        sum_weights = 0
        # For each item in the neighborhood of 'item' i
        for item_j in user_distinct_items:
            # only calculate the weighted values when the weight between item_1 and item_2 exists in weight matrix
            w_item_1_2 = w_matrix[
                (w_matrix["item_1"] == item) & (w_matrix["item_2"] == item_j)
            ]
            if w_item_1_2.shape[0] > 0:
                user_rating_j = user_other_ratings[
                    user_other_ratings[self.item_col] == item_j
                ]
                sum_weighted_other_ratings += (
                    user_rating_j["rating_adjusted"].iloc[0]
                ) * w_item_1_2["weight"].iloc[0]
                sum_weights += np.abs(w_item_1_2["similarity"].iloc[0])

        mean_user_rating = user_other_ratings["rating_adjusted"].iloc[0]

        # if sum_weights is 0 (which may be because of no ratings from new users), use the mean ratings
        if sum_weights == 0:
            predicted_rating = mean_user_rating
        # sum_weights is bigger than 0
        else:
            predicted_rating = (
                mean_user_rating + sum_weighted_other_ratings / sum_weights
            )

        return predicted_rating

    def recommend(self, userID, topn=10):

        w_matrix, adjusted_ratings = self.build_sim_matrix(self.train_df)

        distinct_items = np.unique(adjusted_ratings[self.item_col])
        user_ratings_all_items = pd.DataFrame(columns=[self.item_col, "ratings"])
        user_rating = adjusted_ratings[adjusted_ratings[self.user_col] == userID]

        # calculate the ratings for all items that the user hasn't rated
        i = 0
        for item in distinct_items:
            user_rating = user_rating[user_rating[self.item_col] == item]
            # if the user has rated the item
            if user_rating.shape[0] > 0:
                rating_value = user_ratings_all_items.loc[
                    i, "ratings"
                ] = user_rating.loc[0, item]
            # else predict a rating
            else:
                rating_value = user_ratings_all_items.loc[i, "ratings"] = self.predict(
                    userID, item, w_matrix, adjusted_ratings
                )

            user_ratings_all_items.loc[i] = [item, rating_value]

            i = i + 1

        # select top n items rated by the user
        recommendations = user_ratings_all_items.sort_values(
            by=["ratings"], ascending=False
        ).head(topn)
        return recommendations
