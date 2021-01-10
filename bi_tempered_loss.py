"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.
 Source: https://bit.ly/3jSol8T
 """

import functools
import tensorflow as tf

def for_loop(num_iters, body, initial_args):
    """Runs a simple for-loop with given body and initial_args.
    Args:
      num_iters: Maximum number of iterations.
      body: Body of the for-loop.
      initial_args: Args to the body for the first iteration.
    Returns:
      Output of the final iteration.
    """
    for i in range(num_iters):
        if i == 0:
            outputs = body(*initial_args)
        else:
            outputs = body(*outputs)
    return outputs


def log_t(u, t):
    """Compute log_t for `u`."""

    def _internal_log_t(u, t):
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    return tf.cond(
        tf.math.equal(t, 1.0), lambda: tf.math.log(u),
        functools.partial(_internal_log_t, u, t))


def exp_t(u, t):
    """Compute exp_t for `u`."""

    def _internal_exp_t(u, t):
        return tf.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    return tf.cond(
        tf.math.equal(t, 1.0), lambda: tf.math.exp(u),
        functools.partial(_internal_exp_t, u, t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = tf.math.reduce_max(activations, -1, keepdims=True)
    normalized_activations_step_0 = activations - mu
    shape_normalized_activations = tf.shape(normalized_activations_step_0)

    def iter_body(i, normalized_activations):
        logt_partition = tf.math.reduce_sum(
            exp_t(normalized_activations, t), -1, keepdims=True)
        normalized_activations_t = tf.reshape(
            normalized_activations_step_0 * tf.math.pow(logt_partition, 1.0 - t),
            shape_normalized_activations)
        return [i + 1, normalized_activations_t]

    _, normalized_activations_t = for_loop(num_iters, iter_body,
                                           [0, normalized_activations_step_0])

    logt_partition = tf.math.reduce_sum(
        exp_t(normalized_activations_t, t), -1, keepdims=True)
    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization_binary_search(activations, t, num_iters=10):
    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu = tf.math.reduce_max(activations, -1, keepdims=True)
    normalized_activations = activations - mu
    shape_activations = tf.shape(activations)
    effective_dim = tf.cast(
        tf.math.reduce_sum(
            tf.cast(
                tf.greater(normalized_activations, -1.0 / (1.0 - t)), tf.int32),
            -1,
            keepdims=True), tf.float32)
    shape_partition = tf.concat([shape_activations[:-1], [1]], 0)
    lower = tf.zeros(shape_partition)
    upper = -log_t(1.0 / effective_dim, t) * tf.ones(shape_partition)

    def iter_body(i, lower, upper):
        logt_partition = (upper + lower) / 2.0
        sum_probs = tf.math.reduce_sum(exp_t(
            normalized_activations - logt_partition, t), -1, keepdims=True)
        update = tf.cast(tf.less(sum_probs, 1.0), tf.float32)
        lower = tf.reshape(lower * update + (1.0 - update) * logt_partition,
                           shape_partition)
        upper = tf.reshape(upper * (1.0 - update) + update * logt_partition,
                           shape_partition)
        return [i + 1, lower, upper]

    _, lower, upper = for_loop(num_iters, iter_body, [0, lower, upper])
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return tf.cond(
        tf.less(t, 1.0),
        functools.partial(compute_normalization_binary_search, activations, t,
                          num_iters),
        functools.partial(compute_normalization_fixed_point, activations, t,
                          num_iters))


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    t = tf.convert_to_tensor(t)
    normalization_constants = tf.cond(
        tf.math.equal(t, 1.0),
        lambda: tf.math.log(tf.math.reduce_sum(tf.exp(activations), -1, keepdims=True)),
        functools.partial(compute_normalization, activations, t, num_iters))
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations,
                              labels,
                              t1,
                              t2,
                              label_smoothing=0.0,
                              num_iters=5):
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations.
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1).
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    with tf.name_scope('bitempered_logistic'):
        t1 = tf.convert_to_tensor(t1)
        t2 = tf.convert_to_tensor(t2)
        one = tf.convert_to_tensor(1.0)

        if label_smoothing > 0.0:
            num_classes = tf.cast(tf.shape(labels)[-1], tf.float32)
            labels = (
                             1 - num_classes /
                             (num_classes - 1) * label_smoothing) * labels + label_smoothing / (
                             num_classes - 1)

        @tf.custom_gradient
        def _custom_gradient_bi_tempered_logistic_loss(activations):
            """Bi-Tempered Logistic Loss with custom gradient.
            Args:
              activations: A multi-dimensional tensor with last dim `num_classes`.
            Returns:
              A loss tensor, grad.
            """
            with tf.name_scope('gradient_bitempered_logistic'):
                probabilities = tempered_softmax(activations, t2, num_iters)
                loss_values = tf.math.multiply(
                    labels,
                    log_t(labels + 1e-10, t1) -
                    log_t(probabilities, t1)) - 1.0 / (2.0 - t1) * (
                                      tf.math.pow(labels, 2.0 - t1) - tf.math.pow(probabilities, 2.0 - t1))

                def grad(d_loss):
                    """Explicit gradient calculation.
                    Args:
                      d_loss: Infinitesimal change in the loss value.
                    Returns:
                      Loss gradient.
                    """
                    delta_probs = probabilities - labels
                    forget_factor = tf.math.pow(probabilities, t2 - t1)
                    delta_probs_times_forget_factor = tf.math.multiply(delta_probs,
                                                                       forget_factor)
                    delta_forget_sum = tf.math.reduce_sum(
                        delta_probs_times_forget_factor, -1, keepdims=True)
                    escorts = tf.math.pow(probabilities, t2)
                    escorts = escorts / tf.math.reduce_sum(escorts, -1, keepdims=True)
                    derivative = delta_probs_times_forget_factor - tf.math.multiply(
                        escorts, delta_forget_sum)
                    return tf.math.multiply(d_loss, derivative)

                return loss_values, grad

        loss_values = _custom_gradient_bi_tempered_logistic_loss(activations)

        loss_values = tf.math.reduce_sum(loss_values, -1)

        return loss_values