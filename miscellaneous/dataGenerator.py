from tensorflow import keras
from sklearn.utils import shuffle
import numpy as np
import random
from skimage.io import imread


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
    return np.ceil(len(self.list_img)/self.batch_size).astype("int")

  def __get_image(self, path):
    # Reading the RGB image
    
    data_image = imread(path)
    #selected_bands = [0, 1,2,3, 4]  # Assuming bands are zero-indexed
    #data_image = data_image[:, :, selected_bands]
    data_image = data_image[:,:,:3]
    return data_image

  def __get_label(self, path):
    # Reading the mask
    data_mask = imread(path, as_gray=True)

    mask_max = data_mask.max()
    if mask_max == 255:
      data_mask = data_mask/255

    data_mask = np.expand_dims(data_mask, axis=-1)
    
    return data_mask

  def __getitem__(self, idx):
    i = idx * self.batch_size
    
    current_batch_size = self.batch_size
    if (idx+1) == self.__len__():
      current_batch_size = len(self.list_img[i:])

    # Batch of coordinates
    batch_images = self.list_img[i : i + current_batch_size]
    batch_labels = self.list_label[i : i + current_batch_size]    

    x = np.zeros((current_batch_size, 
                  self.image_size, 
                  self.image_size, 
                  3),
                  dtype=np.uint8)

    y = np.zeros((current_batch_size,
                  self.image_size, 
                  self.image_size,
                  1),
                  dtype=np.bool_)

    for j, (path_image, path_label) in enumerate(zip(batch_images,batch_labels)):
      # Get an individual image and its corresponding label
      x_sample = self.__get_image(path_image)
      y_sample = self.__get_label(path_label)        

      x[j,...] = x_sample      
      y[j,...] = y_sample
    return x, y
