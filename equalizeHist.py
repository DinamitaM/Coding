import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# Upload an image file
print("Please upload an image file:")
uploaded_image = files.upload()

# Extract the filename of the uploaded image
image_filename = next(iter(uploaded_image))
main_image  = cv2.imread(image_filename)

# Upload an image file
print("Please upload an template image file:")
uploaded_template = files.upload()

# Extract the filename of the uploaded image
template_filename  = next(iter(uploaded_template))
template  = cv2.imread(template_filename)

# Convert both images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Enhance contrast using histogram equalization
main_image_eq = cv2.equalizeHist(main_gray)
template_eq = cv2.equalizeHist(template_gray)

w, h = template_gray.shape[::-1]
 
res = cv.matchTemplate(main_image_eq,template_eq,cv.TM_CCOEFF_NORMED)
print(res)
threshold = 0.50
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(main_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
output_filename = 'res.png'
cv.imwrite(output_filename,main_image)
files.download(output_filename)