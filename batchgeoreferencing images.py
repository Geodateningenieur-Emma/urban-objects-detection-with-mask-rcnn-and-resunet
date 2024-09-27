'''
When working with a large image from which to extract buildings or roads, the first step is to split it into 
smaller patches. A simple and efficient method for doing this is to use GDAL Retile, which maintains the 
image's coordinate system. This ensures that your model receives each patch with its name and spatial coordinates, 
allowing it to make predictions and pass the coordinates to the prediction image. 
This approach makes it easier to combine predictions while preserving the spatial integrity of the data.
'''

#Batch georeference images 
import os 
import rasterio

# assume you are working on very large dataset
#sometime you have different cities, countries, for different time, each to be processed individually

set='path to country x data' 
src_dir = 'path to original image with coordinates'
image_paths = []
image_paths1 = []

for filename in os.listdir(src_dir): 
    if os.path.splitext(filename)[1].lower() in ['.tif','.TIF', '.PNG', '.png', '.tiff']:
        image_paths.append(os.path.join(src_dir, filename))

target_dir = f'common path_{set}' 
for filename1 in os.listdir(target_dir):  
    if os.path.splitext(filename1)[1].lower() in ['.tif', '.PNG', '.png', '.tiff']: 
        image_paths1.append(os.path.join(target_dir, filename1))

outpath_georef = f'path where to store output_{set}_georef'
# Ensure the output directory exists
os.makedirs(outpath_georef, exist_ok=True)
ext = ".tif"

for georef_image in image_paths:
    georef_image_name = georef_image.split(os.sep)[-1]
    sID = georef_image_name.split(".")[0]
    print(sID)
    georef = rasterio.open(georef_image)
    
    for predict_image in image_paths1:
        predict_image_name = predict_image.split(os.sep)[-1]
        dsID = predict_image_name.split(".")[0]
        print (dsID)
        
        if dsID == sID:
            toref = cv2.imread(predict_image, 0)
            new_tif = rasterio.open(os.path.join(outpath_georef, sID + ext), 'w',
                                    driver='Gtiff',
                                    height=georef.height,
                                    width=georef.width,
                                    count=1,
                                    nodata=0,
                                    crs=georef.crs,
                                    transform=georef.transform,
                                    dtype='uint8')
            new_tif.write(toref, indexes=1)
            new_tif.close()

    georef.close()

print("Processing complete.")
