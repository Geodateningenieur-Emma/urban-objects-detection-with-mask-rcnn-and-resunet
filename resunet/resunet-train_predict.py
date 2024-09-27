
'''
@ EMMANUEL@igp
The code is used to train a ResUNet model with and without mixup and then use the generated models for ensemble prediction, 
'''

# Check GPU availability and TensorFlow version
import os
import tensorflow as tf
print("TensorFlow version:", tf.version.VERSION)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set the working directory
os.chdir('C:/Users/nyandwi/Desktop/ResUNet')
model_path = '/logs'

# Import necessary modules and libraries
import model_resunet
import numpy as np
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
from tensorflow.keras import layers
import random
from PIL import Image
import shutil
from utils import imgstitch, DatasetLoad

########################### LEARNING RATE SCHEDULER ###########################

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
# def schedlr(epoch, lr):
#     new_lr = 0.001 * (0.1)**(floor(epoch/20))
#     return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 512
BATCH = 8
EPOCHS = 100

#LOSS FUNCTIONS
##############################################################################
"""
 A combination of BCE and DICE
 Vladimir Iglovikov, Selim Seferbekov, Alexander Buslaev, and Alexey Shvets. TernausNetV2: Fully convolutional
network for instance segmentation. In Proceedings of the Conference on Computer Vision and Pattern Recognition
Workshops, pages 233–237, 2018
"""
from keras import backend as K
import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss


def jaccard(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
	# Flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Convert the tensors to binary masks
    y_true_mask = tf.cast(y_true_flat > 0.5, dtype=tf.float32)
    y_pred_mask = tf.cast(y_pred_flat > 0.5, dtype=tf.float32)

    intersection = K.sum(K.abs(y_true_mask * y_pred_mask), axis=-1)
    sum_ = K.sum(K.abs(y_true_mask) + K.abs(y_pred_mask), axis=-1)
    IoU = (intersection +1) / (sum_ - intersection+ 1)
    return IoU

def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Flatten the tensors
    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])
    
    # # Convert the tensors to binary masks
    # y_true= tf.cast(y_true > 0.5, dtype=tf.float32)
    # y_pred= tf.cast(y_pred > 0.5, dtype=tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    focal_tversky = tf.pow((1 - tversky), gamma)

    loss = focal_tversky
    return loss

def bce_ftl_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6, weight_bce=1.0, weight_ftl=1.5):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce_loss = bce(y_true, y_pred)
    ftl_loss = focal_tversky_loss(tf.sigmoid(y_pred), y_true, alpha, beta, gamma, smooth)
    loss = weight_bce * bce_loss + weight_ftl * ftl_loss
    return loss
 

##################################Customised check point to save model with metrics##########################################################
#############################################################################################################################################
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

# Metrics to be used when evaluating the network
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score(num_classes=2, name='f1', average='micro', threshold=0.4)

# Instantiate the network
sgd_optimizer = Adam()

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/20))
    return new_lr

model = model_resunet.ResUNet((IMG_SIZE, IMG_SIZE, 6))
model.summary()
model.compile(optimizer=sgd_optimizer, loss=bce_ftl_loss, metrics=[jaccard, precision, recall, f1])

# Define the checkpoint directory
checkpoint_dir = '.......'

# Define a custom callback to save the model with metrics in the filename

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # Get validation precision, recall, f1, and jaccard values from the logs dictionary
        val_precision_value = logs.get('val_precision_2', 0.0)
        val_recall_value = logs.get('val_recall_2', 0.0)
        val_f1_value = logs.get('val_f1', 0.0)
        val_jaccard_value = logs.get('val_jaccard', 0.0)

        # Format validation metrics with four decimal places
        val_precision_str = f'{val_precision_value:.4f}'
        val_recall_str = f'{val_recall_value:.4f}'
        val_f1_str = f'{val_f1_value:.4f}'
        val_jaccard_str = f'{val_jaccard_value:.4f}'

        # Include validation metrics in the model checkpoint filename
        filename = f'Building_batch8_finetuned_{epoch + 1:02d}_Validation_Precision {val_precision_str}_Recall {val_recall_str}_F {val_f1_str}_Jaccard {val_jaccard_str}.hdf5'
        self.filepath = os.path.join(checkpoint_dir, filename)

        # Save the model with the updated filename
        super().on_epoch_end(epoch, logs=logs)

# Create the custom ModelCheckpoint callback
checkpoint_callback = CustomModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_checkpoint.hdf5'),
    monitor='f1',
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(patience=10, 
                                              monitor='val_f1', 
                                              mode="max", 
                                              restore_best_weights=True)
learningScheduler=LearningRateScheduler(schedlr, verbose=1)


##############################################################################################################################
##################################################Image Generator#############################################################

import os
import glob
from os.path import join, basename
import pandas as pd
from dataGenerator_mixup import DataGenerator as mix  # A generator that implements mixup
from dataGenerator import DataGenerator as original  # A generator that trains on original images without mixup
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the path to your data
PATH_DATA = join("F:", os.sep, "Emmanuel", "data", "building", "train_set")  # Example path

# Collect image and label paths
images = glob.glob(join(PATH_DATA, 'images', '*.tif'))
labels = glob.glob(join(PATH_DATA, 'labels', '*.tif'))

# Create a DataFrame for the images and labels
df = pd.DataFrame()
df["image"] = images
df["labels"] = labels
# You can uncomment and use this line if you want to capture the source of the image
# df["source"] = df["image"].apply(lambda x: basename(x).split(".")[0])

# Define parameters
SEED = 7
batch_size = 32  # Define batch_size explicitly
image_size = (256, 256)  # Define image_size explicitly (replace with your desired size)

# Split data into train, test, and validation sets
df_train, df_test = train_test_split(df,
                                     test_size=0.10,
                                     stratify=None,
                                     random_state=SEED)

df_train, df_val = train_test_split(df_train,
                                    test_size=0.111,  # 10% of the remaining 90% = ~10% of the original data
                                    stratify=None,
                                    shuffle=True,
                                    random_state=SEED)

# Initialize the data generators
train = original(batch_size=batch_size,
                 list_img=df_train["image"].values,
                 list_label=df_train["labels"].values,
                 image_size=image_size,
                 shuffle=True)

val = original(batch_size=batch_size,
               list_img=df_val["image"].values,
               list_label=df_val["labels"].values,
               image_size=image_size,
               shuffle=False)

test = original(batch_size=batch_size,
                list_img=df_test["image"].values,
                list_label=df_test["labels"].values,
                image_size=image_size,
                shuffle=False)

# Fetch a batch of images and masks from the generator
x, y = next(train)  # The generator outputs both x (images) and y (masks) at once

# # Display the first image and mask from the batch
# for i in range(0, 1):  # Change range to display more images if needed
#     image = x[i]
#     mask = y[i]
#     plt.subplot(1, 2, 1)
#     plt.imshow(image[:, :, 0], cmap='gray')  # Adjust channel index based on image format
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask[:, :, 0])  # Adjust mask display format as needed
#     plt.show()

##################################################################################################################################################
########################################Train various model with mixup without mixup##############################################################

# Train
pretrained_weight= '.........'
model = tf.keras.models.load_model(pretrained_weight, custom_objects={'bce_ftl_loss': bce_ftl_loss,'jaccard': jaccard}) # tell which unknown elements should be considered
history = model.fit(train, 
                    validation_data=val,
                    epochs=100, 
                    callbacks=[checkpoint, early_stop]) #, initial_epoch if any 

# Test
scores = model.evaluate(test)
print(model.metrics_names)
print(scores)

################################################################################
######################### apply ensemle to predictions###############################
 
models = []
model_folder = '......'

# List all files with .hdf5 extension in the folder
model_paths = glob.glob(os.path.join(model_folder, '*.hdf5'))

for model_path in model_paths:
    model = tf.keras.models.load_model(model_path, custom_objects={'bce_ftl_loss': bce_ftl_loss, 'jaccard': jaccard})
    models.append(model)

# Define the directory containing the input images
in_images='.............'
# Define the output directory for predictions
out_masks='.............'
os.makedirs(out_masks, exist_ok=True)

# Define the threshold for binary classification
threshold = 0.5

# Expected input image size
expected_image_size = (512, 512)

# Iterate through the image paths
for filename in os.listdir(in_images):
    if os.path.splitext(filename)[1].lower() in ['.png',':TIF' '.PNG', '.tif']:
        image_path = os.path.join(in_images, filename)

        print("Processing image:", image_path)

        imagename = os.path.basename(image_path)
        ID = os.path.splitext(imagename)[0]

        #image = cv2.imread(image_path) # read only image up to four bands  
        image=io.imread(image_path)
        image = image[:, :, [0,2,3]]

        # Check if the input image size matches the expected size
        if image.shape[:2] != expected_image_size:
            print(f"Skipping image {image_path} due to incorrect size. Expected size: {expected_image_size}, Actual size: {image.shape[:2]}")
            continue

        image = np.expand_dims(image, axis=0)

        # Initialize an empty list to store predictions from all models
        all_predictions = []

        # Compute predictions for each model and store them in the list
        for model in models:
            predicted = model.predict(image)[0, :, :, 0]
            all_predictions.append(predicted)

        # Stack predictions from all models along a new axis (channel-wise ensemble)
        ensemble_predictions = np.stack(all_predictions, axis=-1)

        # Take the mean across the ensemble axis
        ensemble_prediction = np.mean(ensemble_predictions, axis=-1)

        # Apply thresholding
        ensemble_prediction[ensemble_prediction > threshold] = 255
        ensemble_prediction[ensemble_prediction <= threshold] = 0

        # Save ensemble_prediction as a binary image
        output_path = os.path.join(out_masks, ID + ".png")
        cv2.imwrite(output_path, ensemble_prediction.astype(np.uint8))

        print("Saved prediction as:", output_path)
