from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
import CAVE as cv
import help_function as hlp_fun

import cv2

show = True
to_restor = False
ifGender = True
heigh_global=56
width_global=56
image_folder = './Images'
image_type = '*.jpg'
batch_size = 100
epochs = 30
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 100
details_dim = 7
num_examples_to_generate = 10
inputs_decoder = 3
dec_in_channels = 3  # RGB - 3 Grayscale - 1

per_distribution = 0.8
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

hf = hlp_fun.help_function()

(train_data, validation_data, test_data, train_details, validation_details, test_details) = hf.get_data(image_folder, image_type, per_distribution)

def preprocess_images(images):
  images = images.reshape(images.shape[0], heigh_global, width_global, 3) / 255
  #images.type()
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

optimizer = tf.keras.optimizers.Adam(1e-4)

#KL-loss
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


def compute_loss(model, x, details):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z, details)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


#@tf.function
def train_step(model, x, details , optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x , details )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = cv.CVAE(latent_dim, details_dim)

def generate_and_save_images(model, epoch, test_sample, details, old_details):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z, details[:,:7])
  for i in range(predictions.shape[0]):
      im = np.array(predictions[i, :, :, :])
      det = np.array(details[i])
      im = im * 255.0
      cv2.imwrite("./images_restor/" + str(i)+"_1_"+str(det[1]) +"_"+ str(det[-1]) + ".jpg",
                  cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

  for i in range(test_sample.shape[0]):
      im = np.array(test_sample[i, :, :, :])
      det = np.array(old_details[i])
      im = im * 255.0
      cv2.imwrite("./images_restor/" + str(i) + "_0_" +str(det[1]) +"_"+ str(det[-1]) + ".jpg",
                  cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

      # plt.subplot(4, 4, i + 1)
      # plt.axis("off")
      # plt.imshow(cv2.cvtColor(np.int32(np.float32(predictions[i, :, :, 0]*255)), cv2.COLOR_BGR2RGB) )

  # tight_layout minimizes the overlap between 2 sub-plots
  # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()
  # Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate


for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
for test_batch_details in test_details_dataset.take(1):
    test_sample_details = test_batch_details[0:num_examples_to_generate, :]
        #generate_and_save_images(model, 0, test_sample, test_sample_details[:,:7])
# train_itar = train_dataset
# details_itar = train_details_dataset.make_initializable_iterator()

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 100


latest = tf.train.latest_checkpoint(checkpoint_dir)

#model.load_weights(latest)


for epoch in range(1, epochs + 1):
    start_time = time.time()
    details_iteration =  iter(train_details_dataset)

    for train_x in train_dataset:

      #print("i: " + str(i))
      train_details_x = details_iteration.get_next()
      train_step(model, train_x, train_details_x[:,:7], optimizer)


    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    details_validation_iteration = iter(validation_details_dataset)
    for validate_batch in validation_dataset:
        validation_details_x = details_validation_iteration.get_next()
        loss(compute_loss(model,validate_batch , validation_details_x[:, :7]))
    elbo = -loss.result()
    print("epoch: " + str(epoch) + " loss: " + str(loss.result()))

    model.save_weights(checkpoint_path.format(epoch=0))




test_itere =  iter(test_details_dataset)
for test_x in test_dataset:
  test_details_x =test_itere.get_next()
  loss(compute_loss(model, test_x,test_details_x[:,:7]))
elbo = loss.result()
display.clear_output(wait=False)
print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
    .format(epoch, elbo, end_time - start_time))
test_sample_details_copy = np.array(test_sample_details)
test_sample_details_copy[:,1] = 1- test_sample_details_copy[:,1]
generate_and_save_images(model, epoch, test_sample, test_sample_details_copy,np.array(test_sample_details) )
