from PIL import Image
import numpy as np
import gym

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from image_processor import ImageProcessor
from custom_models import build_the_model

import matplotlib.pyplot as plt

from collections import deque

import time

train = False



env = gym.make("snake:snake-v0")

nb_actions = env.action_space.n

IMG_SHAPE = (84,84)
WINDOW_LENGTH = 12

input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                               attr='eps',
                               value_max=0.5, # Change this if loading a pre-trained model
                               value_min=0.1,
                               value_test=0.05,
                               nb_steps=1000000)

model = build_the_model(input_shape, nb_actions=nb_actions)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.99,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1)


dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

weights_filename = 'snake_workdir/DQN_BO.h5f'

checkpoint_filename = 'snake_workdir/DQN_CHECKPOINT.h5f'

checkpoint_callback = ModelIntervalCheckpoint(checkpoint_filename, interval=1000)

try:
    model.load_weights(checkpoint_filename)
    print(f"Loaded {checkpoint_filename}")
except:
    print(f"No checkpoint file to load under {checkpoint_filename}")


if train == True:
    metrics = dqn.fit(env, nb_steps=1100, callbacks=[checkpoint_callback], log_interval=10000, visualize=False)
    dqn.test(env, nb_episodes=1, visualize=True)
    env.close()
    model.summary()



observation = env.reset()

for step in range(3000):

    env.render(mode='human')

    observation = processor.process_observation(observation)

    action = dqn.forward(observation)

    observation, reward, done, info = env.step(action)

    if done:
        env.reset()
        done = False

    time.sleep(0.3)

env.close()