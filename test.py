
import tensorflow as tf
import numpy as np

from bi_tempered_loss import bi_tempered_logistic_loss


t1 = 0.8
t2 = 1.4
label_smoothing = 0.01
y_true = tf.convert_to_tensor([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
y_pred = tf.convert_to_tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
loss = bi_tempered_logistic_loss(y_pred, y_true, t1=t1, t2= t2, label_smoothing =label_smoothing)
assert loss.shape == (2,)
print(loss.numpy())

expected = np.array([0.49713564 ,0.92891073])
np.testing.assert_array_almost_equal(loss.numpy(), expected)


