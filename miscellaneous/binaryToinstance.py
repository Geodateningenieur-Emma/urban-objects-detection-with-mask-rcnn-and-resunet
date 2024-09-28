# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:42:08 2023

@author: install
"""

import os
import tifffile as tiff
import numpy as np
from tqdm import tqdm
import cv2
from skimage.io import imread
import random
import tensorflow as tf
from tensorflow import keras

import cv2
import numpy as np
import os

def binary_to_instance_segmentation(binary_image):
    # Label connected components in the binary image
    num_labels, labeled_image = cv2.connectedComponents(binary_image)

    # Create a colormap for visualization (optional)
    colormap = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    # Initialize an empty instance segmentation image
    instance_segmentation = np.zeros_like(binary_image)

    # Assign unique instance IDs to each connected component
    for label in range(1, num_labels):  # Labels start from 1, 0 is background
        instance_mask = (labeled_image == label)
        instance_segmentation[instance_mask] = label

    return instance_segmentation, colormap

def mixup_data(images, masks, alpha=1.0):
    # Generate random indices to shuffle the images and masks
    indices = tf.range(tf.shape(images)[0])
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_images = tf.gather(images, shuffled_indices)
    shuffled_masks = tf.gather(masks, shuffled_indices)
    
    # Generate mixup weight (lambda)
    lam = np.random.beta(alpha, alpha)  # Beta distribution for mixup
    #print(f'lambda: {lam}')
    
    # Convert images and masks to float32
    images = tf.cast(images, tf.float32)
    shuffled_images = tf.cast(shuffled_images, tf.float32)
    masks = tf.cast(masks, tf.float32)
    shuffled_masks = tf.cast(shuffled_masks, tf.float32)
    
    # Apply mixup
    mixedup_images = lam * images + (1.0 - lam) * shuffled_images
    mixedup_masks = tf.maximum(masks, shuffled_masks)
    
    return mixedup_images, mixedup_masks, shuffled_indices