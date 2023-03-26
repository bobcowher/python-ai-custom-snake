from rl.core import Processor
from PIL import Image
import numpy as np


class ImageProcessor(Processor):


    def process_observation(self, observation):
        IMG_SHAPE = (84, 84)
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE)
        img = img.convert("L")
        img = np.array(img)

        return img.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        #         processed_batch = batch.astype('float32')/255.0

        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)