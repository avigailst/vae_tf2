import tensorflow as tf

class graph_model(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, heigh_global, width_global, n_latent, inputs_decoder, reshaped_dim):
      self.heigh_global = heigh_global
      self.width_global = width_global
      self.n_latent = n_latent
      self.inputs_decoder = inputs_decoder
      self.reshaped_dim = reshaped_dim
      self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.heigh_global, self.width_global, 3)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='relu'),
            tf.keras.layers.Dropout(keep_prob, input_shape=(2,)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding= 'same'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='relu'),
            tf.keras.layers.Dropout(.2, input_shape=(2,)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(1, 1), activation='relu'),
            tf.keras.layers.Dropout(.2, input_shape=(2,)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_latent + self.n_latent)
        ]
      )

      self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(self.n_latent,)),
            tf.keras.layers.Dense(units=self.inputs_decoder * 2, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=self.reshaped_dim),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=4, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Dropout(.2, input_shape=(2,)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=4, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Dropout(.2, input_shape=(2,)),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits