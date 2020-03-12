# Load the images in numpy vectors
import glob
import numpy as np
from PIL import Image
from UNet.utils import convert, plot_images

images_path = glob.glob("./results/images/*.png")
labels_path = glob.glob("./results/labels/*.png")
predictions_path = glob.glob("./results/predictions/*.png")

images_list = []
labels_list = []
predictions_list = []

# Group the corresponding image and label together
for image, mask, prediction in zip(images_path, labels_path, predictions_path):
    images_list.append(np.array(Image.open(image)))
    labels_list.append(np.array(Image.open(mask).convert('RGB')))
    predictions_list.append(np.array(Image.open(prediction).convert('RGB')))

images = np.asarray(images_list)
labels = np.asarray(labels_list)
predictions = np.asarray(predictions_list)

# Resize of the images
X = np.asarray(images, dtype=np.float32) / 255
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = np.asarray(labels, dtype=np.float32) / 255
if y.shape[-1] > 1:
    y = y[..., 0]
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

p = np.asarray(predictions, dtype=np.float32) / 255
if p.shape[-1] > 1:
    p = p[..., 0]
p = p.reshape(p.shape[0], p.shape[1], p.shape[2], 1)

plot_images(org_imgs=X, mask_imgs=y, pred_imgs=p, nm_img_to_plot=23, figsize=6)
