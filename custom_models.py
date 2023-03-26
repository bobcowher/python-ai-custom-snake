from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


def build_the_model(input_shape, weights_filename=None, test_run=False, display_summary=False, nb_actions=6):
    print(input_shape)

    model = Sequential()

    model.add(Permute((2, 3, 1), input_shape=input_shape))

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dense(nb_actions))

    model.add(Activation('linear'))

    if weights_filename is not None:
        model.load_weights(weights_filename)

    if test_run:
        model.compile(optimizer='adam', loss='mean_absolute_error')

    if display_summary:
        model.summary()

    return (model)