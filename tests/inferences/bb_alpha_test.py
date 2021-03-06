from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Normal


class test_bb_alpha_class(tf.test.TestCase):

  def _test_normal_normal(self, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([0.0] * 50, dtype=np.float32)

      mu = Normal(loc=0.0, scale=1.0)
      x = Normal(loc=mu, scale=1.0, sample_shape=50)

      qmu_loc = tf.Variable(tf.random_normal([]))
      qmu_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))
      qmu = Normal(loc=qmu_loc, scale=qmu_scale)

      # analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
      inference = ed.BBAlpha({mu: qmu}, data={x: x_data})
      inference.run(*args, **kwargs)

      self.assertAllClose(qmu.mean().eval(), 0, rtol=0.1, atol=0.6)
      self.assertAllClose(qmu.stddev().eval(), np.sqrt(1 / 51),
                          rtol=0.15, atol=0.5)

      variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer')
      old_t, old_variables = sess.run([inference.t, variables])
      self.assertEqual(old_t, inference.n_iter)
      sess.run(inference.reset)
      new_t, new_variables = sess.run([inference.t, variables])
      self.assertEqual(new_t, 0)
      self.assertNotEqual(old_variables, new_variables)

  def _test_model_parameter(self, *args, **kwargs):
    with self.test_session() as sess:
      x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

      p = tf.sigmoid(tf.Variable(0.5))
      x = Bernoulli(probs=p, sample_shape=10)

      inference = ed.BBAlpha({}, data={x: x_data})
      inference.run(*args, **kwargs)

      self.assertAllClose(p.eval(), 0.2, rtol=5e-2, atol=5e-2)

  def test_bb_alpha(self):
    # model parameter - special case - KLqp:
    self._test_normal_normal(n_samples=10, alpha=0.01)

    # model parameter - special case - KLqp:
    self._test_model_parameter(n_samples=10, alpha=0.01)
    # model parameter - case where alpha = 0.5:
    self._test_model_parameter(n_samples=10, alpha=0.5)
    # model parameter - case where alpha > 0:
    self._test_model_parameter(n_samples=10, alpha=5.0)
    # model parameter - case where alpha < 0:
    self._test_model_parameter(n_samples=10, alpha=-5.0)


if __name__ == '__main__':
  ed.set_seed(42)
  tf.test.main()
