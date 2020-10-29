import tensorflow as tf
import numpy as np
import re
import os
import glob
import math
import random
import cv2



class help_function:

    def __init__(self):
        return


    def get_data(self, folder_path, image_type, per_distribution, last_throw = False ):
        files_name = os.path.join(folder_path, image_type)
        # glob goes through all files in dir
        images_name = [file_path for file_path in glob.iglob(files_name)] # List of files name
        random.seed(2019)
        random.shuffle(images_name)
        image_pixel =  np.array([cv2.cvtColor(cv2.imread(file_path, 1),cv2.COLOR_BGR2RGB)for file_path in images_name])
        data = [[file_path,image_pixel[i]] for (i,file_path) in enumerate(images_name)] # List of pixels # Write "mpimg.imread(file_path)[:,:,0]" for one color
        return self.extract_data(data)

    def extract_data(self, data):
        new_data = []
        new_details = []
        for img in data:
            try:
                age_gender_color_by_regex = re.search('([0-9]{1,3})_([0-1])_([0-4])_([0-9]{1,17})', img[0])
                num_of_color = int(age_gender_color_by_regex.group(3).replace("_", ""))
                # if num_of_color!=0 and num_of_color!=1:
                #     continue
                gender = int(age_gender_color_by_regex.group(2).replace("_", ""))
                details = [0] * 8  # Create a list that contains zeros. the number 7 is for [age, gender, color1, color2, color3, color4, color5]
                details[0] = int(age_gender_color_by_regex.group(1).replace("_", ""))
                details[1] = gender
                details[7] = int(age_gender_color_by_regex.group(4).replace("_", ""))
                details[num_of_color + 2] = int(1)
                new_details.append(np.array(details))
                new_data.append(np.array(img[1]))
            except:
                continue
        return  self.split_train_test(new_data, new_details)



    def split_train_test(self, full_dataset,full_details, train_percent = 0.7, validation_percent = 0.15, test_percent = 0.15):


        # Number of features
        dataset_size = len(full_dataset)

        train_size = math.ceil(train_percent * dataset_size)
        test_validation_size = dataset_size - train_size
        validation_size = math.ceil(validation_percent/(1-train_percent) * test_validation_size)

        train_index = train_size
        validation_index = train_index + validation_size

        train_data = full_dataset[: train_index]
        train_details = full_details[: train_index]

        validation_data = full_dataset[train_index : validation_index]
        validation_details = full_details[train_index: validation_index]

        test_data = full_dataset[validation_index :]
        test_details = full_details[validation_index:]

        return train_data, validation_data, test_data, train_details, validation_details, test_details

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
