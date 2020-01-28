import pandas as pd


class PopularityRecommender:

    MODEL_NAME = "Popularity"

    def __init__(self, user_id, item_id, df_lookup=None):
        self.user_id = user_id
        self.item_id = item_id
        self.df_lookup = df_lookup

    @property
    def name(self):
        return self.MODEL_NAME

    def fit(self, train_data, topn=10):

        # Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = (
            train_data.groupby([self.item_id]).agg({self.user_id: "count"}).reset_index()
        )
        train_data_grouped.rename(columns={self.user_id: "score"}, inplace=True)

        # Classement des items et création d'un rank
        train_data_sort = train_data_grouped.sort_values(
            ["score", self.item_id], ascending=[False, True]
        )
        train_data_sort["Rank"] = train_data_sort["score"].rank(
            ascending=0, method="first"
        )

        # top-n recommandations
        self.popularity_recommendations = train_data_sort.head(topn)

    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Ajoute la colonne correspondant à l'utilisateur en question
        user_recommendations["user_id"] = user_id

        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations.merge(self.df_lookup, on=self.item_id, how='inner')
