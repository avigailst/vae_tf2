import matplotlib.pyplot as plt
class generate_images:

  def __init__(self):
    return

  def generate_and_save_images(self,model, epoch, test_sample, details, old_details):
    mean, logvar = model.encode(test_sample, False)
    #z = model.reparameterize(mean, logvar, eps = 0)
    predictions = model.sample(details[:,:7], mean)
    predictions_with_same_details = model.sample(old_details[:,:7], mean)
    p_length = len(predictions)
    fig = plt.figure(figsize=(5, 5))
    for j in range(int(p_length/5)):
      for i in range(5):
        k = i+j*5
        plt.subplot(5, 3, 3*i + 1)
        plt.imshow(predictions[k, :, :, :])
        plt.axis('off')
        plt.subplot(5, 3,3* i + 2)
        plt.imshow(predictions_with_same_details[k, :, :, :])
        plt.axis('off')
        plt.subplot(5, 3,3* i + 3)
        plt.imshow(test_sample[k, :, :, :])
        plt.axis('off')

      # tight_layout minimizes the overlap between 2 sub-plots
      plt.savefig('image_{:04d}.png'.format(j))
      plt.show()

    # for i in range(test_sample.shape[0]):
    #   plt.subplot(5, 5, i + 1)
    #   plt.imshow(test_sample[i, :, :, :])
    #   plt.axis('off')

   # # tight_layout minimizes the overlap between 2 sub-plots
   #  plt.savefig('image_at_epoch_real_{:04d}.png'.format(epoch))
   #  plt.show()