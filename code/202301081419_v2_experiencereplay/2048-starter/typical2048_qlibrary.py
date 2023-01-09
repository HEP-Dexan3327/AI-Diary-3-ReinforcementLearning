import gym
import numpy as np
import tensorflow as tf


def choose(env: gym.Env, _q_values: tf.Tensor, available_dirs: np.ndarray) -> np.ndarray:
    if env.is_full():
        return np.array(-1)
    li = (_q_values[0] * available_dirs).numpy()
    li[available_dirs*1 == 0] = np.nan
    return np.nanargmax(li)


num_inps = 4


def vectorize(_state: tf.Tensor, available_dirs: np.ndarray, type='normal', normalized=True, expand=True) -> tf.Tensor:
    """ turns the state into a vector before feeding into the neural network.
    :param available_dirs: available directions.
    :param _state:         input observations, in a shape (16,)
    :param type:           either 'normal' (default) or 'one-hot'.
                            output shape is (1,16,1) for 'normal', and is (1,16,16=#options) for 'one-hot'.
    """
    if type in ('one-hot', 'one-hot-17'):
        options_per_cell = 17 if type == 'one-hot-17' else 16
        out = tf.math.multiply(tf.expand_dims(tf.one_hot(_state, options_per_cell), 0),
                               available_dirs.reshape((num_inps, 1, 1)))
        if expand:
            out = tf.expand_dims(out, 0)
        if normalized:
            out /= options_per_cell
        return out
    out = tf.math.multiply(tf.expand_dims(_state, 0),
                           available_dirs.reshape((num_inps, 1)))
    if expand:
        out = tf.expand_dims(out, 0)
    return out


def get_input_type(shape: tuple) -> str:
    return 'one-hot' if shape[2] == 16 else \
        'one-hot-17' if shape[2] == 17 else 'normal'
