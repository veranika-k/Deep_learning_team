import skimage
import matplotlib.pyplot as plt
import tifffile as tif
import os
from skimage import io, segmentation, morphology
from skimage.transform import resize
import numpy as np

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior



join = os.path.join
img_size = 512

img_path = "./example_images"
gt_path = "./example_labels"
output_path = "./resized"
os.makedirs(output_path, exist_ok=True)

img_names = sorted(os.listdir(img_path))
gt_names = [img_name.split('.')[0]+'_label.tiff' for img_name in img_names]

for img_name, gt_name in zip(img_names, gt_names):
    # Load image
    if img_name.endswith('.tif') or img_name.endswith('.tiff'):
        img_data = tif.imread(join(img_path, img_name))
    else:
        img_data = io.imread(join(img_path, img_name))
    
    # Convert to grayscale if RGB
    if img_data.ndim == 3:
        img_data = np.mean(img_data, axis=2)  # naive grayscale conversion

    # Resize and expand dims to (512, 512, 1)
    img_resized = resize(img_data, (512, 512), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_resized = img_resized[:, :, np.newaxis]

    # Save as .tif
    out_name = os.path.splitext(img_name)[0] + ".tif"
    out_path_images = join(output_path, "images")
    os.makedirs(out_path_images, exist_ok=True)
    tif.imwrite(join(out_path_images, out_name), img_resized)

    # Resize ground truth the same way if needed
    out_path_labels = join(output_path, "labels")
    os.makedirs(out_path_labels, exist_ok=True)
    gt_data = tif.imread(join(gt_path, gt_name))
    # Conver instance bask to three-class mask: interior, boundary
    interior_map = create_interior_map(gt_data.astype(np.int16))
    gt_resized = resize(interior_map, (512, 512), preserve_range=True, order=0).astype(np.uint8)
    tif.imwrite(join(out_path_labels, f"{img_name.split('.')[0]}_label.tif"), gt_resized)

print('Done')