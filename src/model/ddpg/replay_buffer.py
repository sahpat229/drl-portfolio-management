"""
Source: https://github.com/vermouth1992/deep-learning-playground/blob/master/tensorflow/ddpg/replay_buffer.py
"""
from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0][0] for _ in batch])
        sw_batch = np.array([_[0][1] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4][0] for _ in batch])
        s2w_batch = np.array([_[4][1] for _ in batch])

        return s_batch, sw_batch, a_batch, r_batch, t_batch, s2_batch, s2w_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class ReplayBufferMultiple(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s1, a1, r1, s2, a2, r2, t, s3):
        experience = (s1, a1, r1, s2, a2, r2, t, s3)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1_batch = np.array([_[0][0] for _ in batch])
        s1w_batch = np.array([_[0][1] for _ in batch])
        a1_batch = np.array([_[1] for _ in batch])
        r1_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3][0] for _ in batch])
        s2w_batch = np.array([_[3][1] for _ in batch])
        a2_batch = np.array([_[4] for _ in batch])
        r2_batch = np.array([_[5] for _ in batch])
        t_batch = np.array([_[6] for _ in batch])
        s3_batch = np.array([_[7][0] for _ in batch])
        s3w_batch = np.array([_[7][1] for _ in batch])

        return s1_batch, s1w_batch, a1_batch, r1_batch, s2_batch, s2w_batch, a2_batch, r2_batch, t_batch, s3_batch, s3w_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class ReplayBufferRollout(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, rollout):
        experience = rollout
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1_batch = np.array([exp[0][0] for exp in batch])
        s1w_batch = np.array([exp[0][1] for exp in batch])
        a1_batch = np.array([exp[1] for exp in batch])

        sf_batch = np.array([exp[-1][0] for exp in batch])
        sfw_batch = np.array([exp[-1][1] for exp in batch])
        t_batch = np.array([exp[-2] for exp in batch])

        rs_batch = []
        index = 2
        while index < len(batch[0]):
            rs_batch.append(np.array([exp[index] for exp in batch]))
            index += 4

        return s1_batch, s1w_batch, a1_batch, rs_batch, t_batch, sf_batch, sfw_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0