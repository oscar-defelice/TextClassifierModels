###
### model.py
###
### Created by Oscar de Felice on 23/10/2020.
### Copyright © 2020 Oscar de Felice.
###
### This program is free software: you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation, either version 3 of the License, or
### (at your option) any later version.
###
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###
### You should have received a copy of the GNU General Public License
### along with this program. If not, see <http://www.gnu.org/licenses/>.
###
########################################################################
###
### model.py
### This module contains the model definition
"""
    CNN Model for text classification.
"""

# import libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Embedding, GlobalMaxPool1D, MaxPool1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# default values
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()


def buildModel( vocab_size,
                emb_dim,
                max_len,
                num_classes,
                dropout_rate,
                optimizer = optimizer,
                loss = loss,
                name = 'CNN_for_text_classification'):
    """
        Function to build a CNN model for text classification.

        Parameters
        ----------
            vocab_size : int
                            number of words in the vocabulary.

            emb_dim : int
                        dimension of the embedding space.

            max_len : int
                        maximal length of the input sequences.

            num_classes : int
                            number of unique labels, it is also the number of
                            units of the last dense layer.

            dropout_rate : int
                            dropout hyperparameter, i.e. the probability of dropping
                            a given node in the layer.
                            dropout_rate = 0 is equivalent to no dropout.

            optimizer : optimizer object in Keras
                            default : Adam optimizer

            loss : loss object in Keras
                        default : Categorical Crossentropy

            name : str
                    name of the model.
                    default : 'CNN_for_text_classification'

        Return
        ------
            A Keras model object.

    """
    # build the model

    model = Sequential(name = name)
    model.add(Embedding(vocab_size, output_dim = emb_dim, input_length=max_len))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(50, 3, activation='relu', padding='same', strides=1))
    model.add(MaxPool1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(100, 3, activation='relu', padding='same', strides=1))
    model.add(MaxPool1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(200, 3, activation='relu', padding='same', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss=loss, metrics=['acc'], optimizer=optimizer)
    print(model.summary())

    return model
