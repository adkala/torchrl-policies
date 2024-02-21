from torchrl.data import TensorDictReplayBuffer, BoundedTensorSpec
from torchrl.data.replay_buffers.storages import LazyTensorStorage

import time


class PrintLogger:
    def __init__(self, name):
        self.step = 0

    def log_scalar(self, key, value, step=None):
        if step and step > self.step:
            self.step = step
            print(f"\nStep {step}")
        print(f"{key}: {value}")


def create_space_spec(low, high, shape, dtype, device=None):
    return BoundedTensorSpec(low, high, shape=shape, device=device, dtype=dtype)


def create_replay_buffer(buffer_size=1_000_000, pin_memory=False):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=pin_memory,
        storage=LazyTensorStorage(
            buffer_size,
        ),
    )
    return replay_buffer


def get_time_str():
    return time.strftime("%d-%m-%Y_%H-%M-%S")
