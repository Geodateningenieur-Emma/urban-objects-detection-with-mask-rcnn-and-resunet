"""
Created on Wed Apr 26 09:59:16 2023
# @author: Emmanuel@IGP, 
# This code is improved by Zhang (https://github.com/gitWayneZhang/Mask_RCNN).
# Most importantly I customized this code to make run on large-scale remote image,
# You can use it for detecting and counting buildings, tree crowns, wind turbines, people, ....with geotiffs as inputs and georeferenced outputs.
# Along with this code, I attached a trained model file.
##############################################################################
"""
# import required libraries
import os
import sys
import glob
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import pandas as pd
from osgeo import gdal
import rasterio

# Root directory of the project (assume your mask rcnn folder is placed on C:/)
os.chdir('C:/MaskRCNNTF2') # 
ROOT_DIR = os.getcwd()
print(ROOT_DIR)

# point to the subfolder that contains the model, utils, ....
MRCNN_PATH= os.path.abspath(os.path.join(ROOT_DIR, 'mrcnn'))  
sys.path.append(MRCNN_PATH) # to find local version of library

#Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_table
import mrcnn.model as modellib
from mrcnn.model import log
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#print (sys.path)
# print (tf.__version__)
# print (np.__version__)
# print (tf.keras.__version__)

# model configuration: hyperparameters, dataloaders, etc
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class RwandaConfig(Config):
    """Configuration for training.Derives from the base Config class and overrides values specific to the COCO dataset
    The following are few Hyper-parameters specific to Mask R-CNN (https://medium.com/analytics-vidhya/taming-the-hyper-parameters-of-mask-rcnn-3742cb3f0e1b)
    (1)The Backbone:the Conv Net architecture that is to be used in the first step of Mask R-CNN. The available options for choice of Backbones include ResNet50, ResNet101
    (2)Train_ROIs_Per_Image: This is the maximum number of ROI’s, the Region Proposal Network will generate for the image, which will further be processed for classification and masking in the next stage. 
       The ideal way is to start with default values if number of instances in the image are unknown. If the number of instances are limited, it can be reduced to reduce the training time
    (3) Max_GT_Instances: This is the maximum number of instances that can be detected in one image. This helps in reduction of false positives and reduces the training time
    (4) Detection_Min_Confidence:confidence level threshold, beyond which the classification of an instance will happen. If accuracy of detection is important, increase the threshold to ensure that 
       there are minimal false positive by guaranteeing that the model predicts only the instances with very high confidence
    (5)Image_Min_Dim and Image_Max_Dim.
    (6) Loss weights:
        Mask RCNN uses a complex loss function which is calculated as the weighted sum of different losses at each and every state of the model. The loss weight hyper parameters corresponds to the weight that the model should assign to each of its stages.
        Rpn_class_loss: This corresponds to the loss that is to assigned to improper classification of anchor boxes (presence/absence of any object) by Region proposal network. This should be increased when multiple objects are not being detected 
        by the model in the final output. Increasing this ensures that region proposal network will capture it.
        Rpn_bbox_loss: This corresponds to the localization accuracy of the RPN. This is the weight to tune in case, the object is being detected but the bounding box should be corrected.
        Mrcnn_class_loss: This corresponds to the loss that is assigned to improper classification of object that is present in the region proposal. This is to be increased in case the object is being detected from the image, but misclassified
        Mrcnn_bbox_loss: This is the loss, assigned on the localization of the bounding box of the identified class, It is to be increased if correct classification of the object is done, but localization is not precise. 
        Mrcnn_mask_loss: This corresponds to masks created on the identified objects, If identification at pixel level is of importance, this weight is to be increased
    (7)The batch size defines the number of samples that will be propagated through the network. determine the amount of information before weights and bias are updated
    OTHER IMPORTANT PARAMETERS. IMAGES_PER_GPU, RPN_ANCHOR_SCALES, STEPS_PER_EPOCH,VALIDATION_STEPS 
    """
    # Give the configuration a recognizable name
    NAME = "Rwanda"
    BACKBONE='resnet101'  # two options here resnet 50, or 101 but other vesion can be used by modification of core code
    BATCH_SIZE=4
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) # ajust depending on the size of target objects 
    GPU_COUNT = 1 #default is 1
    DETECTION_MIN_CONFIDENCE = 0.7  # tune to see changes 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Rwanda building
    IMAGE_MAX_DIM=512
    IMAGE_MIN_DIM=512
    STEPS_PER_EPOCH = 50  # Total Number of Training Samples) / Batch Size
    VALIDATION_STEPS = 50 # TotalvalidationSamples / ValidationBatchSize
    
config = RwandaConfig()
print (config.IMAGE_SHAPE)
config.display()

############################################################################################
# Define data loader
# We use images and label rasters (0 background  and 1, 2, .... n instances for each patch)
# No need to strugle comverting in coco style or html format
class RwandaDataset(utils.Dataset):
    def load_Rwanda(self, dataset_dir):
        """Load a subset of the Rwanda dataset.
        dataset_dir: The root directory of the training dataset.
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
         Load a subset of the Rwanda dataset from the given dataset_dir.
        """
        # Add classes
        self.add_class("Rwanda", 1, "1")
        #loading images
        self._image_dir = os.path.join(dataset_dir, "images/")    # RGB images
        self._mask_dir = os.path.join(dataset_dir, "labels/")     # labels images
        i=0
        for f in glob.glob(os.path.join(self._image_dir, "*.tif")):
            filename = os.path.split(f)[1]
            self.add_image("Rwanda", image_id=i, path=f,
                          width=config.IMAGE_SHAPE[0], height=config.IMAGE_SHAPE[1], filename=filename)
            i += 1
            
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        """Read instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        self._mask_dir = os.path.join(dataset_dir, "labels")
        info = self.image_info[image_id]
        fname = info["filename"]
        masks = []
        class_ids = []
        #looping through all the classes, loading and processing corresponding masks
        m_src = skimage.io.imread(os.path.join(self._mask_dir, "1", fname))     # '/TrainDataset/labels/1'       
        #making individual masks for each instance
        instance_ids = np.unique(m_src)
        #print (instance_ids)
        for i in instance_ids:
            if i > 0:
                m = np.zeros(m_src.shape)
                m[m_src==i] = i
                #print(i)
                if np.any(m==i):
                    masks.append(m)
        # class_ids.append(class_id)
        try:
            masks = np.stack(masks, axis=-1)        
        except:
            print("!There were no mask.", info)
            
        # Return mask, and array of class IDs of each instance.
        return masks.astype(np.bool), np.ones([masks.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return a link to the image dataset."""
        info = self.image_info[image_id]
        if info["source"] == "Rwanda":
            return info["Rwanda"]
        else:
            super(self.__class__).image_reference(self, image_id)

###############################################################################
# Load datasets
dataset_train = RwandaDataset()
dataset_train.load_Rwanda(os.path.join(ROOT_DIR, 'annotation_dataset/finetuning/train'))
dataset_train.prepare()

#sanity check:
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:40}".format(i, info['name']))
image_ids = np.random.choice(dataset_train.image_ids,61)  # sample of images to display 
print (image_ids)
for image_id in image_ids:
    print (image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    #print (class_ids)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names,limit=1)

# Sometimes you will run into an error, out of index data: poorly digitized (touching or overlapping) instances,  
# Ensure overlapping errors are corrected after digitization and use PIL to read labels and remove wrong labels
# in our experiment those images cannot be read by PIL, so can be easily detected and removed.

dataset_val = RwandaDataset()
dataset_val.load_Rwanda(os.path.join(ROOT_DIR, 'annotation_dataset/finetuning/val'))
dataset_val.prepare()

# Must call before using the dataset
print("Image Count: {}".format(len(dataset_val.image_ids))) # check if you are able to read image
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))

##############################################################################

# Create model in training mode
model = modellib.MaskRCNN(mode="training", 
                          config=config,
                          model_dir=os.path.join(ROOT_DIR,'logs'))

# Which weights to start with?
#coco=''
spacenet = ""
last= 'add path here/pre-trainedmodel.h5'

init_with = "last"  # imagenet, coco, last, spacenet

if init_with == "coco":
    # Load weights trained on MS COCO
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "spacenet":
    #Load weights trained on spanenet dataset
    model.load_weights(spacenet, by_name=True),
                        #exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                #"mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(last, by_name=True) # model.find_last(), by_name=True
model.keras_model.summary()

#tf.keras.utils.plot_model(model.keras_model, to_file="temp.png", show_shapes=True)

#Explore various training strategy: set layers to: '4+', 'heads', 'all', use augmentation,     
#ADD CUSTOMISED_CALLBACKS
# from keras import callbacks
# custom_callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=10, custom_callbacks=[custom_callbacks], 
#             layers='heads')
## train the head layers 
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=120, 
            layers='heads')

##train  from fith layers
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE*2, 
#             epochs=120, 
#             layers='4+')
##train with augmentation
# import imgaug
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE/10, 
#             epochs=70, 
#             layers='heads',
#             augmentation = imgaug.augmenters.Sequential([ 
#                 imgaug.augmenters.Fliplr(1), 
#                 imgaug.augmenters.Flipud(1), 
#                 imgaug.augmenters.Affine(rotate=(-45, 45)), 
#                 imgaug.augmenters.Affine(rotate=(-90, 90)), 
#                 imgaug.augmenters.Affine(scale=(0.5, 1.5))]))
history = model.keras_model.history.history
model.keras_model.save_weights("add path here/trained.h5")

# ##############################################################################

# # visualize training 
# epochs = range(1, len(history['loss'])+1)
# plt.figure(figsize=(10,10))
# plt.plot(epochs, history["loss"], label="Train loss")
# plt.plot(epochs, history["val_loss"], label="Valid loss")
# plt.legend()
# plt.show()

##############################################################################

# GET THE READY THE PREDICTION MODEL 
class InferenceConfig(RwandaConfig):
    BATCH_SIZE=8
    #GPU_COUNT = 1
    #IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=os.path.join(ROOT_DIR,'logs'))
# Get path to saved weights
# Either set a specific path or find last trained weights
TRAIN_WEIGHTS_PATH='path to trained model file'
model.load_weights(TRAIN_WEIGHTS_PATH, by_name=True)

########################################EVALUATE #####################################
# calculate the mAP for a model on a given dataset (based on IoU of bboxs), I use this option less 
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle

def evaluate_model(dataset_val, model, inference_config):
 	APs = list()
 	for image_id in dataset_val.image_ids:
		# load image, bounding boxes and masks for the image id
 	  image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
 	  scaled_image = mold_image(image, inference_config)
		# convert image into one sample
 	  sample = expand_dims(scaled_image, 0)
		# make prediction
 	  yhat = model.detect(sample, verbose=0)
		# extract results for first sample
 	  r = yhat[0]
		# calculate statistics, including AP
 	  AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
 	  APs.append(AP)
 	# calculate the mean AP across all images
 	mAP = mean(APs)
 	return mAP
evaluate_model(dataset_val, model, inference_config)

# EVALUATE THE MODEL USING  THE mAP AND A GIVEN  DATASET BASED IoU of MASKS
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import merged_mask
from mrcnn.utils import compute_iou_2
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle

def evaluate_model_IoU_Masks(dataset_val, model, inference_config):
    APs = list()
  # load image, bounding boxes and masks for the image id
    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, inference_config) # convert pixel values (e.g. center)
        sample = expand_dims(scaled_image, 0) 	# convert image into one sample
        yhat = model.detect(sample, verbose=0) # extract results for first sample	
        predict_mask = yhat[0]["masks"]
        AP= compute_iou_2(predict_mask, gt_mask) # calculate statistics, including AP
        APs.append(AP)
     	# calculate the mean AP across all images
    mAP = mean(APs)
    return  mAP
evaluate_model_IoU_Masks(dataset_val, model, inference_config)

# #####################################MAKE PREDICTION ##########################################
#PREDICTION ON BATCH OF IMAGES
#Loop through images, grab image name(ID, grab geographical/projected i.e map space coordinates, predict and generate reference buildings)
import skimage
from skimage.util import img_as_ubyte
import rasterio
import matplotlib.pyplot as plt

real_test_dir = 'D:/images2019/kigali/tiles' # use same size as training image size. 
image_paths = []

for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.tif', '.tiff']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    #print (image_path)
    imagename = image_path.split(os.sep)[-1]
    #print(imagename)
    ID=imagename.split(".")[0]
    print(ID)
    imagesource=rasterio.open(image_path)
    
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],dataset_val.class_names, r['scores'], figsize=(5,5))
    # extract the masks
    masks = results[0]['masks']
    # convert boolean value to int
    mask = np.any(masks.astype(np.bool), axis=-1)
    mask = mask.astype(int)
    mask=mask*255
    # display masks
    plt.figure(figsize=(8,8))
    plt.imshow(mask)

 # add coordinates to mask
    imagesource.crs
    outpath='D:/images2019/kigali/detection'
    ext=".tif"
    new_tif = rasterio.open(os.path.join(outpath, ID + ext),'w',
                            driver='Gtiff',
                            height = imagesource.height,
                            width = imagesource.width,
                            count = 1,
                            nodata=0,
                            crs = imagesource.crs,
                            transform = imagesource.transform,
                            dtype = 'uint8')
    new_tif.write(mask, 1) #result from calculations

    imagesource.close()
    new_tif.close()  
        
#convert to shaepefiles 

from osgeo import gdal, ogr, osr
import os
# import geopandas as gpd
# import fiona
in_path = 'D:/images2019/kigali/detection'
out_path = 'D:/images2019/kigali/shp'
final_path='D:/images2019/kigali/final'
for f in os.listdir(in_path):
    #print(f)
    f_name=f.split(".")[0]
    #print(f_name)
    f_path=os.path.join(in_path, f_name + '.tif')
    #print(f_path)
    shp_path=os.path.join(out_path, f_name + '.shp')
    shp_final_path=os.path.join(final_path, f_name + '.shp')
    # # #  get raster datasource
    src_ds = gdal.Open( f_path)
    srcband = src_ds.GetRasterBand(1)
    dst_layername = f_name
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( shp_path )
    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput('EPSG:4326')
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )
    fld = ogr.FieldDefn("Class", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("Class")
    gdal.Polygonize( srcband, None, dst_layer, dst_field, [], callback=None )
    del src_ds
    del dst_ds
    
# Merge all shapefiles

from pathlib import Path
import pandas as pd
import geopandas as gpd
import os 

# Set the directory containing the shapefiles

directory = 'PATH TO SHAPEFILES'

# Get a list of all the shapefiles in the directory
shapefiles = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.shp')]

# Load the first shapefile as a GeoDataFrame
gdf = gpd.read_file(shapefiles[0])

# Loop through the remaining shapefiles and merge them into the GeoDataFrame
for shapefile in shapefiles[1:]:
    gdf_to_merge = gpd.read_file(shapefile)
    gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf_to_merge], ignore_index=True))

# Write the merged GeoDataFrame to a new shapefile
gdf.to_file('SPECIFY PATH', driver='ESRI Shapefile')

#select by attribute and delete class=0, the background.
#select by attribute and save polygon withc class=255 except class=0, the background.
import geopandas as gpd
import fiona 
dataSrc = gpd.read_file('PATH WHERE PRE-FINAL DATA ARE TO BE SAVED')
dataSrc[dataSrc['Class']==255].to_file('SPECIFY PATH')

#DISSOLVE BOUNDARIES 
#dissolve boundaries
shapes=gpd.read_file('PATH WHERE DATA ARE TO BE SAVED')
dissolved_shapes=shapes.dissolve(by='Class')
dissolved_shapes.to_file('SPECIFY PATH', driver='ESRI Shapefile')

#MULTIPART TO SINGLE PART 
multiPart=gpd.read_file('SPECIFY PATH')
singlePart=multiPart.explode()
singlePart.to_file('SPECIFY PATH')
