###
### train.py
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
### train.py
### This module contains the script to train the model via CLI.
"""
    To train the model this type in a command line prompt
    the command

        python3 -m train [-h] [-c CONFIG]

    where CONFIG is the yaml file path containing configuration variables.
"""

import argparse
from model import buildModel
from utils import getTokeniser, loadConfig, loadData, splitData, tokenise, printReport

# let user feed in 1 parameter, the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, dest='config',
                    help='path of the configuration file')
args = parser.parse_args()
configFile = args.config

## load configuration and store it in a dictionary
config = loadConfig(configFile)

## variables
# path of csv file containing data
input_data = config['dataset']['file']['path']
# split mode (random or fixed) and value (test_rate or validation column in df)
split_mode = config['trn_val_splits']['type']
split_value = config['trn_val_splits']['value']
# random state seed for pseudorandom processes
random_state = config['random_state']
# name of the column containing text data
text_col = config['input_features']['text']
# name of the column containing labels
labels = config['input_features']['labels']
# tokeniser object
tokeniser_conf = config['tokeniser']['tokeniser']
tokeniser = getTokeniser(tokeniser_conf)
# max sequence length
max_len = config['tokeniser']['max_seq_length']
# pad to max_seq_length
pad = config['tokeniser']['pad']
# embedding dimension
embedding_dim = config['module']['embedding_dim']
# dropout rate
dropout_rate = config['module']['dropout_rate']
# batch size
batch_size = config['training']['batch_size']
# number of epochs
n_epochs = config['training']['epochs']
# learning rate
learning_rate = config['training']['lr']
# output file for the report
outfile = config['output']['path']


## load data
df, n_classes = loadData(input_data, labels)

## train, test split
df_train, df_test = splitData(df, split_mode, split_value, random_state)

## text-to-sequence encoding
X_train, vocab_size = tokenise(tokeniser, df_train, text_col, max_len,
                                padding = pad, mode = 'train')
X_test = tokenise(tokeniser, df_test, text_col, max_len, padding = pad,
                    mode = 'test')

## convert labels to one-hot encoder
y_train = encodeLabels(df_train, labels)
y_test = encodeLabels(df_test, labels)

## convert to tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
ds_train = train_dataset.shuffle(10000).batch(batch_size)
ds_test = test_dataset.batch(batch_size)


## define deep learning model calssifier
model = buildModel(vocab_size, embedding_dim, max_len, n_classes, dropout_rate)

## train the model
model.fit(ds_train, epochs=n_epochs, validation_data=ds_test)

printReport(model, X_test, y_test, target_names = df[labels].unique(),
            outfile=outfile)
