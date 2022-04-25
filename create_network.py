import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, SimpleRNN, Dense, Conv1D, MaxPool1D, LSTM,
                                     BatchNormalization, Dropout, Activation, GlobalMaxPooling1D, GRU,
                                     MultiHeadAttention, Flatten)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
import pandas as pd


def create_network(outs,
                   vocab_size,
                   output_dim,
                   len_max,
                   dense_layers,
                   attention_layers,
                   activation=None,
                   activation_dense=None,
                   lambda_regularization=None,
                   dropout=None,
                   lrate=0.0001):

    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # Initialize the model
    model = Sequential()

    # Adds an embedding layer
    # Input_dim = size of the vocabulary
    # Output_dim = length of the vector for each word (essentially a hyperparameter)
    # input_length = maximum length of a sequence
    model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=len_max))

    model.add(Conv1D(filters=64, kernel_size=25, activation='relu', strides=2, padding='same'))


   # Embedding/Compression/Input into MHA layer, then in the end have several dense layers
    # MHA taking examples/time/# of channels -> output of same shape
    # Then flatten and taking additional steps from there
    # You can also just tell the MHA what the output shape should be, so it can do that on its own
    # In MHA you can skip the RNN, figure out what time steps are important, and then what out is a representation of that
    # Maybe just have flatten? Apparently works fine?
    # Information at time 1 needs to be at same level as at time k (What does that mean?)
    # With RNN we were taking info from only final time step, attention takes all of them


    # TODO: Figure out what shapes should be here, and what should be input for parameters
    # One tensor for keys and queries, one for values
    for i in range(len(attention_layers)):
        model.add(MultiHeadAttention(num_heads=attention_layers[i]['heads'], key_dim=output_dim))

    model.add(Flatten())

    for i in range(len(dense_layers)):
        model.add(Dense(units=dense_layers[i]['units'],
                        activation=activation_dense,
                        use_bias=True,
                        kernel_initializer='random_uniform',
                        kernel_regularizer=lambda_regularization))

        if dropout:
            model.add(Dropout(rate=dropout))

    # Turn into a pandas dataframe in order to use nunique to find appropriate number of output units.
    outs = pd.DataFrame(outs)
    model.add(Dense(units=outs.nunique(),
                    use_bias=True,
                    kernel_initializer='random_uniform',
                    activation='softmax',
                    name='Output_layer',
                    kernel_regularizer=lambda_regularization))

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                   epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])


    return model


# An attempt at declaring the order of layers at the command line.
# Takes in the layer type and the parameters associated with the layer, and outputs the layer
#TODO: Figure out how to ensure the parameters match the given layer.
# Maybe a stack of tuples?

def makeLayer(type, parameters):
    if type == 'LSTM':
        layer = (LSTM(parameters['n_neurons'],
                      activation=parameters['tanh'],
                      use_bias=True,
                      return_sequences=False,  # Produce entire sequence of outputs
                      kernel_initializer='random_uniform',
                      kernel_regularizer=lambda_regularization,
                      unroll=False))

    if type == 'RNN':
        layer = (SimpleRNN(parameters['n_neurons'],
                           activation=parameters['tanh'],
                           use_bias=True,
                           return_sequences=False,  # Produce entire sequence of outputs
                           kernel_initializer='random_uniform',
                           kernel_regularizer=lambda_regularization,
                           unroll=False))

    if type == 'CNN':
        layer = (Conv1D(filters=64, kernel_size=25, activation='relu'))

    # Not used in this assignment, but can be potentially recycled
    if type == 'MP':
        layer = MaxPooling2D(pool_size=(3,3),
                              strides=(1,1),
                              padding='same')

    if type == 'Dense':
        layer = (Dense(parameters['n_neurons'],
                           activation=parameters['tanh'],
                           use_bias=True,
                           return_sequences=False,  # Produce entire sequence of outputs
                           kernel_initializer='random_uniform',
                           kernel_regularizer=lambda_regularization,
                           unroll=False))


    return layer

