from keras import backend
import tensorflow as tf
import numpy as np

# We assume that we are working with square images of size DIMS x DIMS
DIMS = 224


def mse_visible(y_true, y_pred):
    """
    Mean squared error that takes into the account only the data points that are visible in the image
    :param y_true: array of true data points of shape (batch_size, 2*n_points)
    :param y_pred: array of predicted data points of shape (batch_size, 2*n_points)
    :return:
    """
    # Check for each coordinate in y_pred if it is in the bounds
    in_bounds = tf.math.logical_and(y_true >= 0, y_true < DIMS)

    # Permute the columns to change x and y coordinates. This is needed to check whether both of these coordinates are
    # in bounds, otherwise the point is not visible
    permutation = [i+1 if i % 2 == 0 else i-1 for i in range(y_pred.shape[1])]
    in_bounds_permuted = tf.gather(in_bounds, indices=permutation, axis=1)

    in_bounds = tf.math.logical_and(in_bounds, in_bounds_permuted)
    in_bounds = tf.dtypes.cast(in_bounds, dtype="float32")

    # Calculate the squared difference between predictions and true values
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.math.squared_difference(y_pred, y_true)
    # Multiply the difference with in_bounds mask (so the coordinates that are not visible get multiplied by 0
    diff = diff * in_bounds

    # Compute the number of visible coordinates and divide each instance by that (we average over only the visible
    # coordinates)
    weights = tf.math.reduce_sum(in_bounds, axis=1)
    weights = tf.reshape(weights, (-1, 1))
    diff = diff / weights

    # As we already weighted the entries we only sum them together to get the average
    return backend.sum(diff, axis=1)


def mse_visible_lists(y_true, y_pred):
    """
    Does the same as mse_visible, except that the inputs are now lists of predictions and true values for only one
    instance
    :param y_true: list - true keypoints
    :param y_pred: list - predicted keypoints
    :return: float - average squared error over the coordinates of visible keypoints
    """
    visible_diffs = [(x_p - x_t)**2 + (y_p - y_t)**2 for (x_t, y_t, x_p, y_p) in
                     zip(y_true[::2], y_true[1::2], y_pred[::2], y_pred[1::2])]

    return np.mean(visible_diffs) / 2
