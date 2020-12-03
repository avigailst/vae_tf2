from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
# import tensorflow_probability as tfp
import time
import os
import  generate_images as gen_im
import loss_function as lo_fu
import help_function as hlp_fun
import cvae as cv

hf = hlp_fun.help_function()
lf = lo_fu.loss_function()
gi = gen_im.generate_images()

# train_size = 60000
batch_size = 100
# test_size = 10000
latent_dim = 200
num_examples_to_generate = 25
epochs = 5
image_folder = './Images'
image_type = '*.jpg'
per_distribution = 0.9
heigh_global=56
width_global=56
details_dim = 7
to_load = False
learning_rate = 1e-4

(train_data, validation_data, test_data, train_details, validation_details, test_details) = hf.get_data(image_folder, image_type, per_distribution)

def preprocess_images(images):
  images = images.reshape(images.shape[0], heigh_global, width_global, 3) / 255.
  return images.astype('float32')

train_images = preprocess_images(np.array(train_data))
validation_images = preprocess_images(np.array(validation_data))
test_images = preprocess_images(np.array(test_data))


train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .batch(batch_size))
validation_dataset = (tf.data.Dataset.from_tensor_slices(validation_images)
                 .batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .batch(batch_size))
train_details_dataset = (tf.data.Dataset.from_tensor_slices(train_details)
                 .batch(batch_size))
validation_details_dataset = (tf.data.Dataset.from_tensor_slices(validation_details)
                 .batch(batch_size))
test_details_dataset = (tf.data.Dataset.from_tensor_slices(test_details)
                .batch(batch_size))

optimizer = tf.keras.optimizers.Adam(learning_rate)


checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)

for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
for test_batch_details in test_details_dataset.take(1):
    test_sample_details = test_batch_details[0:num_examples_to_generate, :]

#@tf.function
def train_step(model, x, details, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = lf.compute_loss(model, x, details)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


  # set the dimensionality of the latent space to a plane for visualization later

  # keeping the random vector constant for generation (prediction) so
  # it will be easier to see the improvement.

random_vector_for_generation = tf.random.normal(
      shape=[num_examples_to_generate, latent_dim])
model = cv.CVAE(latent_dim,details_dim)
if to_load:
    model.load_weights(latest)

  # Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate

epoch = 0
for epoch in range(1, epochs + 1):
  start_time = time.time()
  details_iteration = iter(train_details_dataset)
  for train_x in train_dataset:
      train_details_x = details_iteration.get_next()
      train_step(model, train_x, train_details_x[:, :7], optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  details_validation_iteration = iter(validation_details_dataset)
  for validate_batch in validation_dataset:
      validation_details_x = details_validation_iteration.get_next()
      loss(lf.compute_loss(model, validate_batch, validation_details_x[:, :7]))
  elbo = -loss.result()
  print("epoch: " + str(epoch) + " loss: " + str(loss.result()))
  model.save_weights(checkpoint_path.format(epoch=0))

test_itere =  iter(test_details_dataset)
for test_x in test_dataset:
  test_details_x =test_itere.get_next()
  loss(lf.compute_loss(model, test_x,test_details_x[:,:7]))
elbo = loss.result()
display.clear_output(wait=False)
print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
    .format(epoch, elbo, end_time - start_time))
test_sample_details_copy = np.array(test_sample_details)
test_sample_details_copy[:,1] = 1- test_sample_details_copy[:,1]
gi.generate_and_save_images(model, epoch, test_sample, test_sample_details_copy,np.array(test_sample_details) )