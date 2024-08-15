import cv2
from google.colab import files
import numpy as np
from matplotlib import pyplot as plt

# Upload an image file
print("Please upload an image file:")
uploaded_image = files.upload()

# Extract the filename of the uploaded image
image_filename = next(iter(uploaded_image))
main_image  = cv2.imread(image_filename)

# Upload an image file
print("Please upload an morning template image file:")
uploaded_template_morning0 = files.upload()

# Extract the filename of the uploaded image
template_filename_morning0  = next(iter(uploaded_template_morning0))
template_morning0  = cv2.imread(template_filename_morning0)

# Upload an image file
print("Please upload an morning template image file:")
uploaded_template_morning1 = files.upload()

# Extract the filename of the uploaded image
template_filename_morning1  = next(iter(uploaded_template_morning1))
template_morning1  = cv2.imread(template_filename_morning1)

# Upload an image file
print("Please upload an morning template image file:")
uploaded_template_morning2 = files.upload()

# Extract the filename of the uploaded image
template_filename_morning2  = next(iter(uploaded_template_morning2))
template_morning2  = cv2.imread(template_filename_morning2)

# Upload an image file
print("Please upload an morning template image file:")
uploaded_template_morning3 = files.upload()

# Extract the filename of the uploaded image
template_filename_morning3  = next(iter(uploaded_template_morning3))
template_morning3  = cv2.imread(template_filename_morning0)

# Convert both images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_morning0_gray = cv2.cvtColor(template_morning0, cv2.COLOR_BGR2GRAY)
template_morning1_gray = cv2.cvtColor(template_morning1, cv2.COLOR_BGR2GRAY)
template_morning2_gray = cv2.cvtColor(template_morning2, cv2.COLOR_BGR2GRAY)
template_morning3_gray = cv2.cvtColor(template_morning3, cv2.COLOR_BGR2GRAY)

# Enhance contrast using histogram equalization
main_image_eq = cv2.equalizeHist(main_gray)
template_morning0_eq = cv2.equalizeHist(template_morning0_gray)
template_morning1_eq = cv2.equalizeHist(template_morning0_gray)
template_morning2_eq = cv2.equalizeHist(template_morning0_gray)
template_morning3_eq = cv2.equalizeHist(template_morning0_gray)

w_morning0, h_morning0 = template_morning0_gray.shape[::-1]
w_morning1, h_morning1 = template_morning1_gray.shape[::-1]
w_morning2, h_morning2 = template_morning2_gray.shape[::-1]
w_morning3, h_morning3 = template_morning3_gray.shape[::-1]
 
res_morning0 = cv2.matchTemplate(main_image_eq,template_morning0_eq,cv2.TM_CCOEFF_NORMED)
res_morning1 = cv2.matchTemplate(main_image_eq,template_morning1_eq,cv2.TM_CCOEFF_NORMED)
res_morning2 = cv2.matchTemplate(main_image_eq,template_morning2_eq,cv2.TM_CCOEFF_NORMED)
res_morning3 = cv2.matchTemplate(main_image_eq,template_morning3_eq,cv2.TM_CCOEFF_NORMED)

threshold = 0.65
loc = np.where( res_morning0 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_morning0, pt[1] + h_morning0), (0,0,255), 2)

loc = np.where( res_morning1 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_morning1, pt[1] + h_morning1), (0,0,255), 2)

loc = np.where( res_morning2 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_morning2, pt[1] + h_morning2), (0,0,255), 2)

loc = np.where( res_morning3 >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_morning3, pt[1] + h_morning3), (0,0,255), 2)

output_filename = 'res.png'
cv2.imwrite(output_filename,main_image)
files.download(output_filename)