import tensorflow as tf

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim,details_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.details_dim = details_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(56, 56, 3)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=1, padding='same'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim + details_dim,)),
            tf.keras.layers.Dense(units=14*14*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(14, 14, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=4, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=4, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Dropout(.5),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=2, strides=1, padding='same',
                activation='relu'),

        ]
    )

 # @tf.function
  def sample(self, eps=None, details=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, details)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z,details,  apply_sigmoid=True):
    logits = self.decoder(tf.concat([z, tf.cast(details,'float32')], 1))
 #   logits = tf.types.cast(logits,tf.float64)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits