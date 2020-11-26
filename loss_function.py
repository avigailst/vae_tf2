import tensorflow as tf
import numpy as np
class loss_function:

    def __init__(self):
        return

    def log_normal_pdf(self,sample, mean, logvar, raxis=1):
      log2pi = tf.math.log(2. * np.pi)
      return tf.reduce_sum(
          -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)


    def compute_loss(self,model, x, details):
      mean, logvar = model.encode(x, True)
      z = model.reparameterize(mean, logvar)
      x_logit = model.decode(z, details)
      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
      logpz = self.log_normal_pdf(z, 0., 0.)
      logqz_x = self.log_normal_pdf(z, mean, logvar)
      return -tf.reduce_mean(logpx_z + logpz)# - logqz_x)