###
### utils.py
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
### utils.py
### This module contains helper functions for train and model.

# import libraries
import tensorflow_datasets as tfds
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import re
import string
from datetime import datetime

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def getTokeniser(tokeniser_conf):
    if tokeniser_conf == 'keras':
        from keras.preprocessing.text import Tokenizer
        return Tokenizer(lower = True)
    else:
        raise ValueError(f'{tokeniser_conf} is not a valid option for tokeniser')

def loadConfig(configFile):
    with open(configFile, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)

def loadData(data, label_col, **kwargs):
    """
        Function to download and store data into a dataset.

        Parameters
        ----------
            data : str
                    path to data file (csv).

            label_col : str
                            name of the column containing labels.

        Returns
        -------
            pandas dataframe containing data.
            n_classes : int
                            nuber of classes for text classification.
    """
    df = pd.read_csv(data, **kwargs)
    return df, len(df[label_col].unique())

def splitData(  df, split_mode, split_value, random_state = None):
    """
        Function to split data into train and validation sets.

        Parameters
        ----------
            df : pandas dataframe
                    dataframe to be splitted

            split_mode : str
                            string indicating whether the split is random or
                            based on the value in a specific column

            split_value : float or str
                            if split is random, which fraction of df has to be
                            taken as validation set.
                            if split is value based, the name of the column to
                            look at.

            random_state : int
                            seed to recover reproducibility in pseudorandom
                            operations.

        Return
        ------
            df_train, df_test : pandas dataframes
                                    pandas dataframes containing train and test
                                    data respectively.
    """

    if split_mode == 'random':
        df_train, df_test = train_test_split(   df, test_size=split_value,
                                                random_state=random_state)
    elif split_mode == 'fixed':
        test_mask = df[split_value] == 'validation'
        df_test = df[test_mask]
        df_train = df[~test_mask]

    else:
        raise ValueError(f'{split_mode} is not a valid option for split_mode.')

    return df_train, df_test

def remove_punc(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    """
        Functions to remove stopwords.
    """
    list1 = [word for word in text.split() if
                word not in stopwords.words('english')]

    return " ".join(list1)

def preprocess( df, text_col):
    """
        Function to preprocess text data.

        Parameters
        ----------
            df : pandas dataframe
                    dataframe containing text data.

            text_col : str
                        column name containing text to be transformed.

        Returns
        -------
            NoneType, it updates df[text_col] with stopwords and punctuation
            removed.
    """

    df[text_col] = df[text_col].apply(lambda x: remove_punc(x))
    df[text_col] = df[text_col].apply(lambda x: remove_stopwords(x))


def tokenise(   tokeniser,
                df,
                text_col,
                max_len,
                padding = True,
                mode = 'train'):
    """
        Function to operate the text-to-sequence conversion.

        Parameters
        ----------
            tokeniser : tokeniser object

            df : pandas dataframe
                    dataframe containing text data in a column.

            text_col : str
                        name of the column containing text to tokenise.

            max_len : int
                        maximal lenght of the tokenised sequence.
                        Texts in text_col longer than max_len are truncated.
                        Shorter ones are padded with special token.

            padding : bool
                        Set to True to add pad tokens to sequences shorter than
                        max_len.
                        default : True

            mode : str
                    train mode indicates to operate the tokeniser fit on
                    sequences.
                    test mode just convert text to sequences.

        Return
        ------
            numpy array of shape (len(df), max_len)
            This contains a numerical sequence per row corresponding to the
            encoding of each df[text_col] row.
            In mode train returns also the vocab_size.
    """
    preprocess(df, text_col)

    if mode == 'train':
        tokeniser = tokeniser
        tokeniser.fit_on_texts(df[text_col])
        vocab_size = len(tokeniser.word_index) + 1
    elif mode == 'test':
        tokeniser = tokeniser
    else:
        raise ValueError(f'{mode} is not a valid option.')

    tokenised_texts = tokeniser.texts_to_sequences(df[text_col])
    if padding:
        tokenised_texts = pad_sequences(tokenised_texts, maxlen=max_len)

    if mode == 'train':
        return tokenised_texts, vocab_size
    else:
        return tokenised_texts

def encodeLabels(df, label_col, mode):
    """
        Function to apply the one-hot encoder to labels.

        Parameters
        ----------
            df : pandas dataframe
                    dataframe containing data.

            label_col : str
                            name of the column containing labels.
    """

    encoded_labels = preprocessing.LabelEncoder()

    y = encoded_labels.fit_transform(df[label_col])
    
    return to_categorical(y)

def printReport(model, x_test, y_test,
                target_names = None,
                num_digits = 4,
                outfile = None):
    """
        Function to print classification report.

        Parameters
        ----------
            y_true : list of float
                        the test labels.

            y_predict : list of float
                            model predictions for labels.

            label_names : list of str
                            list of names for labels.
                            If None there will appear numbers (indices
                            of label list).
                            default : None

            num_digits : int
                            the number of digits to show in the report.
                            default : 4

            outfile : str or NoneType
                        A path to a file .txt to be filled with classification
                        report.
                        If None, prints on screen.
                        default : None

    """
    y_pred = to_categorical(np.argmax(model.predict(X_test), axis=1))

    if outfile != None:
        original_stdout = sys.stdout # Save a reference to the original standard output

        filename = outfile + datetime.now() + '.txt'

        with open(filename, 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(classification_report(y_test, y_pred,
                                        target_names=label_names,
                                        digits = num_digits))

            sys.stdout = original_stdout # Reset the standard output to its
                                         # original value.

    else:
        print(classification_report(y_test, y_pred, target_names=label_names,
                                    digits = num_digits))
