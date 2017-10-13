from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy, get_descendants

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class BB_alpha(VariationalInference):
  """Variational inference with the Black Box alpha Divergence

  It minimizes the reparameterized ELBO
  
  $\\text{KL}( q(z; \lambda) \| p(z \mid x) )
  - \frac{1}{\alpha} \sum_n \log \mathbb{E}_{q(z; \lambda)} [f_n(\omega)^{\alpha}]$

  This class minimizes the objective by automatically selecting from
  black box inference techniques.

  """
  def __init__(self, *args, **kwargs):
    super(BB_alpha, self).__init__(*args, **kwargs)

  def initialize(self, n_samples=1, alpha=0.5, kl_scaling=None, *args, **kwargs):
    """Initialize inference algorithm. It initializes hyperparameters
    and builds ops for the algorithm's computation graph.
    Args:
        n_samples: int, optional.
            Number of samples from variational model for calculating
            stochastic gradients.
        alpha: float, optional.
            Black box alpha divergence coefficient.
    """
    if kl_scaling is None:
      kl_scaling = {}

    self.n_samples = n_samples
    self.alpha = alpha
    self.kl_scaling = kl_scaling
    return super(BB_alpha, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
    """Wrapper for the `BB_alpha` loss function.

    $-\\text{ELBO} =
    -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]
    - \frac{1}{\alpha} \sum_n \log \mathbb{E}_{q(z; \lambda)} [ f_n(\omega)^{\alpha} ]$

    BB_alpha supports reparameterization gradients [@kingma2014auto]
    of the loss function.

    If the KL divergence between the variational model and the prior
    is tractable, then the loss function can be written as

    $\\text{KL}( q(z; \lambda) \| p(z) )
    - \frac{1}{\alpha} \sum_n \log \mathbb{E}_{q(z; \lambda)} [ f_n(\omega)^{\alpha} ]$

    where the KL term is computed analytically [@kingma2014auto]. We
    compute this automatically when $p(z)$ and $q(z; \lambda)$ are
    Normal.
    
    Notes
        + If the model is not reparameterizable, it returns a
        NotImplementedError.
    """
    is_reparameterizable = all([
        rv.reparameterization_type ==
        tf.contrib.distributions.FULLY_REPARAMETERIZED
        for rv in six.itervalues(self.latent_vars)])
    is_analytic_kl = all([isinstance(z, Normal) and isinstance(qz, Normal)
                          for z, qz in six.iteritems(self.latent_vars)])
    if not is_analytic_kl and self.kl_scaling:
      raise TypeError("kl_scaling must be None when using non-analytic KL term")
    if is_reparameterizable:
      if is_analytic_kl:
        return build_reparam_bb_alpha_kl_loss_and_gradients(self, var_list, self.alpha)
      else:
        return build_reparam_bb_alpha_loss_and_gradients(self, var_list, self.alpha)
    else:
      raise NotImplementedError(
          "Black box alpha divergence inference only works with reparameterizable"
          " models.")


def build_reparam_bb_alpha_kl_loss_and_gradients(inference, var_list, alpha):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  $ -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
          + \\text{KL}(q(z; \lambda) \| p(z)) ) $

  based on the reparameterization trick [@kingma2014auto].

  It assumes the KL is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_lik = [0.0] * inference.n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(inference.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value()

    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_lik[s] += tf.reduce_sum(
            inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  kl_penalty = tf.reduce_sum([
      tf.reduce_sum(inference.kl_scaling.get(z, 1.0) * kl_divergence(qz, z))
      for z, qz in six.iteritems(inference.latent_vars)])

  if np.abs(alpha - 0.0) < 10e-3:
    p_log_lik = tf.reduce_mean(p_log_lik)
    loss = p_log_lik - kl_penalty
  
  else:
    p_log_lik = tf.stack(p_log_lik)
    p_log_lik = p_log_lik * alpha
    p_log_lik_max = tf.reduce_max(p_log_lik, 0)
    p_log_lik = tf.log(
        tf.maximum(1e-9,
             tf.reduce_mean(tf.exp(p_log_lik - p_log_lik_max), 0)))
    p_log_lik = (p_log_lik + p_log_lik_max) / alpha
    p_log_lik = tf.reduce_mean(p_log_lik)
    loss = p_log_lik - kl_penalty
    
  
  if inference.logging:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=[inference._summary_key])

  loss = -loss

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars

def build_reparam_bb_alpha_loss_and_gradients(inference, var_list, alpha):
  """Build loss function. Its automatic differentiation
  is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  p_log_prob = [0.0] * inference.n_samples
  q_log_prob = [0.0] * inference.n_samples
  p_log_lik = [0.0] * inference.n_samples
  base_scope = tf.get_default_graph().unique_name("inference") + '/'
  for s in range(inference.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
    scope = base_scope + tf.get_default_graph().unique_name("sample")
    dict_swap = {}
    for x, qx in six.iteritems(inference.data):
      if isinstance(x, RandomVariable):
        if isinstance(qx, RandomVariable):
          qx_copy = copy(qx, scope=scope)
          dict_swap[x] = qx_copy.value()
        else:
          dict_swap[x] = qx

    for z, qz in six.iteritems(inference.latent_vars):
      # Copy q(z) to obtain new set of posterior samples.
      qz_copy = copy(qz, scope=scope)
      dict_swap[z] = qz_copy.value()
      q_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))

    for z in six.iterkeys(inference.latent_vars):
      z_copy = copy(z, dict_swap, scope=scope)
      p_log_prob[s] += tf.reduce_sum(
          inference.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))

    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(
            inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
        
    for x in six.iterkeys(inference.data):
      if isinstance(x, RandomVariable):
        x_copy = copy(x, dict_swap, scope=scope)
        p_log_lik[s] += tf.reduce_sum(
            inference.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)

  if np.abs(alpha - 0.0) < 10e-3:
    p_log_lik = tf.reduce_mean(p_log_lik)
    loss = p_log_prob - q_log_prob
    
  else:
    p_log_lik = tf.stack(p_log_lik)
    p_log_lik = p_log_lik * alpha
    p_log_lik_max = tf.reduce_max(p_log_lik, 0)
    p_log_lik = tf.log(
        tf.maximum(1e-9,
             tf.reduce_mean(tf.exp(p_log_lik - p_log_lik_max), 0)))
    p_log_lik = (p_log_lik + p_log_lik_max) / alpha
    p_log_lik = tf.reduce_mean(p_log_lik)
    loss = p_log_prob - q_log_prob + p_log_lik
    
  if inference.logging:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[inference._summary_key])
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=[inference._summary_key])

  loss = -loss

  grads = tf.gradients(loss, var_list)
  grads_and_vars = list(zip(grads, var_list))
  return loss, grads_and_vars