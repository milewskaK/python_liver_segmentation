import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.color import rgb2gray
from skimage import measure
from skimage import *
import cv2 as cv
from skimage.feature import peak_local_max
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label

url = "Patient01.mha"
data_image = sitk.ReadImage(url)
max_index = data_image.GetDepth()
img_t1 = sitk.ReadImage(url)
nda = sitk.GetArrayFromImage(img_t1)
nda1 = nda.ravel()
print(nda)
plt.show(nda)

list_of_2D_images = [data_image[:,:,i] for i in range(max_index)]
list_of_2D_images_np = [sitk.GetArrayFromImage(data_image[:,:,i]) for i in range(max_index)]
print(len(list_of_2D_images_np))

image = list_of_2D_images_np[15]
print(image.shape)
liver1_gray = rgb2gray(liver_image_1)
plt.imshow(liver1_gray)
plt.show()

image_id = image.ravel()
hist,bins = np.histogram(image_id, bins = np.arange(256))
image = (image*255).astype(np.uint8)

ret,thres = cv.threshold(image, 100,240,cv.THRES_OTSU)

#EDGE DETECTION - CANNY
edges = cv.Canny(thres, 25,120)

#find maxima
coordinates = peak_local_max(image, min_distance = 20)

#regions
label_image = label(edges)

for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    circ = mpatches.Circle(9minc, minr, maxc - minc, maxr - minr, fill = False, edgecolor = 'red', linewidth = 2)
    add_patch(circ)

plt.tight_layout()

#region growing
seed = (100,400)
thres = 15
seed_inte = image[seed]
uth = seed_inte + thres
dth = seed_inte - thres
thresholded_image = np.logical_and(image < uth, image > dth)
labeled_image = measure.label(thresholded_image, background = 0)
output = np.zeros(image.shape)
output[labeled_image == labeled_image[seed]] = 1

cv.imshow('Threshold', thres)
cv.imshow('Canny', edges)

plt.figure(1)
plt.plot(bins[:-1], hist, lw=2, c='k')
plt.imshow(output_image)
plt.show()

plt.figure(2)
plt.imshow(image, cmap = 'gray')
plt.plot(coordinates[:, 1], coordinates[:, 0], c = 'r.')
plt.title('regions')
plt.axis('off')

plt.figure(4)
plt.imshow(output, cmap = 'gray')
plt.title('region grow')
plt.show()
