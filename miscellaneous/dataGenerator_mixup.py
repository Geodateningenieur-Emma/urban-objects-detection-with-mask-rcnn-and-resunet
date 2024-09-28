#Define a method within your DataGenerator class, such as mixup_data, to apply mixup to your batch of data. 
#You can call this method from the __getitem__ method before returning the data. Here's how you can do it:

from tensorflow import keras
from sklearn.utils import shuffle
import numpy as np
import random
from skimage.io import imread
import tensorflow as tf
import numpy as np

def mixup_segmentation(images, masks, alpha=1.0):
    # Generate random indices to shuffle the images and masks
    indices = tf.range(tf.shape(images)[0])
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_images = tf.gather(images, shuffled_indices)
    shuffled_masks = tf.gather(masks, shuffled_indices)
    
    # Generate image weight (minimum 0.4 and maximum 0.6)
    lam = tf.random.uniform([], minval=0.4, maxval=0.6, dtype=tf.float32)  # Convert to float32
    print(f'lambda: {lam}')
    
    # Convert images to float32
    images = tf.cast(images, tf.float32)
    shuffled_images = tf.cast(shuffled_images, tf.float32)
    
    # Weighted Mixup
    mixedup_images = lam * images + (1.0 - lam) * shuffled_images
    mixedup_masks = lam * masks + (1.0 - lam) * shuffled_masks
    
    return mixedup_images, mixedup_masks, shuffled_indices

# def mixup_segmentation(images, masks, alpha=1.0):
#     # Generate random indices to shuffle the images and masks
#     indices = tf.range(tf.shape(images)[0])
#     shuffled_indices = tf.random.shuffle(indices)
#     shuffled_images = tf.gather(images, shuffled_indices)
#     shuffled_masks = tf.gather(masks, shuffled_indices)

#     # Convert images to float32
#     images = tf.cast(images, tf.float32)
#     shuffled_images = tf.cast(shuffled_images, tf.float32)
    
#     # Generate image weight (minimum 0.0 and maximum 1.0)
#     lam = tf.random.uniform([], minval=0.1, maxval=0.4, dtype=tf.float32)  # Convert to float32
#     print(f'lambda: {lam}')
    
#     # Weighted Mixup for images
#     mixedup_images = lam * images + (1.0 - lam) * shuffled_images

#     # Mixup for masks (union)
#     mixedup_masks = tf.maximum(masks, shuffled_masks)
    
#     return mixedup_images, mixedup_masks, shuffled_indices


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
               batch_size,                              
               list_img,                
               list_label,
               image_size,            
               shuffle=True):
        self.batch_size = batch_size    
        self.list_img = list_img    
        self.list_label = list_label
        self.image_size = image_size
        self.shuffle= shuffle    

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.list_img, self.list_label))
            random.shuffle(temp)
            self.list_img, self.list_label = zip(*temp)      

    def __len__(self):
        return int(np.ceil(len(self.list_img) / self.batch_size))

    def __get_image(self, path):
        # Reading the RGB image
        data_image = imread(path)
        data_image = data_image[:,:,:6]
        return data_image

    def __get_label(self, path):
        # Reading the mask
        data_mask = imread(path, as_gray=True)

        mask_max = data_mask.max()
        if mask_max == 255:
            data_mask = data_mask/255

        data_mask = np.expand_dims(data_mask, axis=-1)

        return data_mask

    def mixup_data(self, x, y):
    # Apply mixup to the batch of data (x and y)
       mixed_x, mixed_y, _ = mixup_segmentation(x, tf.cast(y, tf.float32), alpha=1.0)
       return mixed_x, mixed_y

    def __getitem__(self, idx):
        i = idx * self.batch_size

        current_batch_size = self.batch_size
        if (idx + 1) == self.__len__():
            current_batch_size = len(self.list_img[i:])

        # Batch of coordinates
        batch_images = self.list_img[i: i + current_batch_size]
        batch_labels = self.list_label[i: i + current_batch_size]

        x = np.zeros((current_batch_size,
                      self.image_size,
                      self.image_size,
                      6),
                      dtype=np.uint8)

        y = np.zeros((current_batch_size,
                      self.image_size,
                      self.image_size,
                      1),
                      dtype=np.bool_)

        for j, (path_image, path_label) in enumerate(zip(batch_images, batch_labels)):
            # Get an individual image and its corresponding label
            x_sample = self.__get_image(path_image)
            y_sample = self.__get_label(path_label)

            x[j, ...] = x_sample
            y[j, ...] = y_sample

        # Optionally apply mixup to the batch of data
        if self.shuffle:
            x, y = self.mixup_data(x, y)

        return x, y



# import glob

# images = glob.glob ('C:/RESUNET/data/uav_zanzibar/image/*.png')
# labels = glob.glob('C:/RESUNET/data/uav_zanzibar/mask/*.png')             
# batch_size = 8
# image_size = 512

# train = DataGenerator(batch_size=batch_size, 
#                      list_img=images,
#                      list_label=labels,
#                      image_size=image_size,
#                      shuffle=True )

# images=glob.glob('data/val/images/image/*.png')
# labels=glob.glob('data/val/masks/mask/*.png')             
# batch_size=8

# val = DataGenerator(batch_size=batch_size, 
#                    list_img= images,
#                    list_label= labels,
#                    image_size=image_size,
#                    shuffle=False)

# import matplotlib.pyplot as plt

# # For sanity check
# for x,y in train:
#   print(x.shape, y.shape)
#   for x_i,y_i in zip(x,y):
#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow(x_i)

#     plt.subplot(1,2,2)
#     plt.imshow(y_i)
#     plt.show()
#   break