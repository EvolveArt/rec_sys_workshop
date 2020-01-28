#####
# Author :  KevienLiao159
# https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master/movie_recommender/movie_recommendation_using_NeuMF.ipynb
####


import os

# data science imports
import numpy as np
import pandas as pd


# keras/tensorflow imports
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Multiply,
    Concatenate,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

# visualization imports
import matplotlib.pyplot as plt


class GMF:
    def __init__(self, num_users, num_items, latent_dim, vu_reg, vi_reg):
        self.num_items = num_items
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.vu_reg = vu_reg
        self.vi_reg = vi_reg

    def build_model(self):
        """
        Build Generalized Matrix Factorization Model Topology
        
        Parameters
        ----------
        num_users: int, total number of users
        num_iterms: int, total number of items
        latent_dim: int, embedded dimension for user vector and item vector
        vu_reg: float, L2 regularization of user embedded layer
        vi_reg: float, L2 regularization of item embedded layer

        Return
        ------
        A Keras Model with GMF model architeture
        """
        # Input variables
        user_input = Input(shape=(1,), dtype="int32", name="user_input")
        item_input = Input(shape=(1,), dtype="int32", name="item_input")

        MF_Embedding_User = Embedding(
            input_dim=self.num_users + 1,
            output_dim=self.latent_dim,
            embeddings_initializer="uniform",
            name="user_embedding",
            embeddings_regularizer=l2(self.vu_reg),
            input_length=1,
        )
        MF_Embedding_Item = Embedding(
            input_dim=self.num_items + 1,
            output_dim=self.latent_dim,
            embeddings_initializer="uniform",
            name="item_embedding",
            embeddings_regularizer=l2(self.vi_reg),
            input_length=1,
        )

        # Crucial to flatten an embedding vector!
        user_latent = Flatten()(MF_Embedding_User(user_input))
        item_latent = Flatten()(MF_Embedding_Item(item_input))

        # Element-wise product of user and item embeddings
        predict_vector = Multiply()([user_latent, item_latent])

        # Final prediction layer
        prediction = Dense(1, kernel_initializer="glorot_uniform", name="prediction")(
            predict_vector
        )

        # Stitch input and output
        model = Model([user_input, item_input], prediction)

        return model


class MLP:
    def __init__(self, num_users, num_items, layers, reg_layers):
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.reg_layers = reg_layers

    def build_model(self):
        """
        Build Multi-Layer Perceptron Model Topology
        
        Parameters
        ----------
        num_users: int, total number of users
        num_iterms: int, total number of items
        layers: list of int, each element is the number of hidden units for each layer,
            with the exception of first element. First element is the sum of dims of
            user latent vector and item latent vector
        reg_layers: list of int, each element is the L2 regularization parameter for
            each layer in MLP

        Return
        ------
        A Keras Model with MLP model architeture
        """
        assert len(self.layers) == len(self.reg_layers)
        num_layer = len(self.layers)  # Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype="int32", name="user_input")
        item_input = Input(shape=(1,), dtype="int32", name="item_input")

        MLP_Embedding_User = Embedding(
            input_dim=self.num_users + 1,
            output_dim=self.layers[0] // 2,
            embeddings_initializer="uniform",
            name="user_embedding",
            embeddings_regularizer=l2(self.reg_layers[0]),
            input_length=1,
        )
        MLP_Embedding_Item = Embedding(
            input_dim=self.num_items + 1,
            output_dim=self.layers[0] // 2,
            embeddings_initializer="uniform",
            name="item_embedding",
            embeddings_regularizer=l2(self.reg_layers[0]),
            input_length=1,
        )

        # Crucial to flatten an embedding vector!
        user_latent = Flatten()(MLP_Embedding_User(user_input))
        item_latent = Flatten()(MLP_Embedding_Item(item_input))

        # The 0-th layer is the concatenation of embedding layers
        vector = Concatenate(axis=-1)([user_latent, item_latent])

        # MLP layers
        for idx in range(1, num_layer):
            layer = Dense(
                units=self.layers[idx],
                activation="relu",
                kernel_initializer="glorot_uniform",
                kernel_regularizer=l2(self.reg_layers[idx]),
                name="layer%d" % idx,
            )
            vector = layer(vector)

        # Final prediction layer
        prediction = Dense(1, kernel_initializer="glorot_uniform", name="prediction")(
            vector
        )

        # Stitch input and output
        model = Model([user_input, item_input], prediction)

        return model


class Trainer:
    def __init__(self, learner, batch_size, epochs, val_split, inputs, outputs):
        self.learner = learner
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.inputs = inputs
        self.outputs = outputs

    def train_model(self, model, data_path="models/"):
        """
        define training routine, train models and save best model
        
        Parameters
        ----------
        model: a Keras model
        learner: str, one of ['sgd', 'adam', 'rmsprop', 'adagrad']
        batch_size: num samples per update
        epochs: num iterations
        val_split: split ratio for validation data
        inputs: inputs data
        outputs: outputs data
        """
        # add customized metric
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_true - y_pred)))

        # compile model
        model.compile(
            optimizer=self.learner.lower(),
            loss="mean_squared_error",
            metrics=["mean_squared_error", rmse],
        )

        # add call backs
        early_stopper = EarlyStopping(monitor="val_rmse", patience=10, verbose=1)
        model_saver = ModelCheckpoint(
            filepath=os.path.join(data_path, "tmp/model.hdf5"),
            monitor="val_rmse",
            save_best_only=True,
            save_weights_only=True,
        )
        # train model
        history = model.fit(
            self.inputs,
            self.outputs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.val_split,
            callbacks=[early_stopper, model_saver],
        )
        return history

    @staticmethod
    def load_trained_model(model, weights_path):
        model.load_weights(weights_path)
        return model

    @staticmethod
    def plot_learning_curve(history, metric):
        """
        Plot learning curve to compare training error vs. validation error
        """
        # get training error
        errors = history.history[metric]
        # get validation error
        val_errors = history.history["val_{}".format(metric)]
        # get epochs
        epochs = range(1, len(errors) + 1)

        # plot
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, errors, "bo", label="training {}".format(metric))
        plt.plot(epochs, val_errors, "b", label="validation {}".format(metric))
        plt.xlabel("number of epochs")
        plt.ylabel(metric)
        plt.title("Model Learning Curve")
        plt.grid(True)
        plt.legend()
        plt.show()


class NeuMF:
    def __init__(self, num_users, num_items, MF_dim, MF_reg, MLP_layers, MLP_regs):
        self.num_items = num_items
        self.num_users = num_users
        self.MF_dim = MF_dim
        self.MF_reg = MF_reg
        self.MLP_layers = MLP_layers
        self.MLP_regs = MLP_regs

    def build_model(self):
        """
        Build Neural Matrix Factorization (NeuMF) Model Topology.
        This is stack version of both GMF and MLP
        
        Parameters
        ----------
        num_users: int, total number of users
        num_iterms: int, total number of items
        MF_dim: int, embedded dimension for user vector and item vector in MF
        MF_reg: tuple of float, L2 regularization of MF embedded layer
        MLP_layers: list of int, each element is the number of hidden units for each MLP layer,
            with the exception of first element. First element is the sum of dims of
            user latent vector and item latent vector
        MLP_regs: list of int, each element is the L2 regularization parameter for
            each layer in MLP

        Return
        ------
        A Keras Model with MLP model architeture
        """
        assert len(self.MLP_layers) == len(self.MLP_regs)
        num_MLP_layer = len(self.MLP_layers)  # Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype="int32", name="user_input")
        item_input = Input(shape=(1,), dtype="int32", name="item_input")

        # Embedding layer

        # MF
        MF_Embedding_User = Embedding(
            input_dim=self.num_users + 1,
            output_dim=self.MF_dim,
            embeddings_initializer="uniform",
            name="mf_user_embedding",
            embeddings_regularizer=l2(self.MF_reg[0]),
            input_length=1,
        )
        MF_Embedding_Item = Embedding(
            input_dim=self.num_items + 1,
            output_dim=self.MF_dim,
            embeddings_initializer="uniform",
            name="mf_item_embedding",
            embeddings_regularizer=l2(self.MF_reg[1]),
            input_length=1,
        )

        # MLP
        MLP_Embedding_User = Embedding(
            input_dim=self.num_users + 1,
            output_dim=self.MLP_layers[0] // 2,
            embeddings_initializer="uniform",
            name="mlp_user_embedding",
            embeddings_regularizer=l2(self.MLP_regs[0]),
            input_length=1,
        )
        MLP_Embedding_Item = Embedding(
            input_dim=self.num_items + 1,
            output_dim=self.MLP_layers[0] // 2,
            embeddings_initializer="uniform",
            name="mlp_item_embedding",
            embeddings_regularizer=l2(self.MLP_regs[0]),
            input_length=1,
        )

        # MF part
        mf_user_latent = Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
        mf_vector = Multiply()([mf_user_latent, mf_item_latent])

        # MLP part
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
        mlp_vector = Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
        for idx in range(1, num_MLP_layer):
            layer = Dense(
                units=self.MLP_layers[idx],
                activation="relu",
                kernel_initializer="glorot_uniform",
                kernel_regularizer=l2(self.MLP_regs[idx]),
                name="layer%d" % idx,
            )
            mlp_vector = layer(mlp_vector)

        # Concatenate MF and MLP parts
        predict_vector = Concatenate(axis=-1)([mf_vector, mlp_vector])

        # Final prediction layer
        prediction = Dense(1, kernel_initializer="glorot_uniform", name="prediction")(
            predict_vector
        )

        # Stitch input and output
        model = Model([user_input, item_input], prediction)

        return model
