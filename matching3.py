import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files

# Upload an image file
print("Please upload an image file:")
uploaded_image = files.upload()

# Extract the filename of the uploaded image
image_filename = next(iter(uploaded_image))
main_image  = cv2.imread(image_filename)

# Upload an image file
print("Please upload an morning template image file:")
uploaded_template_morning = files.upload()

# Extract the filename of the uploaded image
template_filename_morning  = next(iter(uploaded_template_morning))
template_morning  = cv2.imread(template_filename_morning)

# Upload an image file
print("Please upload an afternoon template image file:")
uploaded_template_afternoon = files.upload()

# Extract the filename of the uploaded image
template_filename_afternoon  = next(iter(uploaded_template_afternoon))
template_afternoon  = cv2.imread(template_filename_afternoon)

# Upload an image file
print("Please upload an night template image file:")
uploaded_template_night = files.upload()

# Extract the filename of the uploaded image
template_filename_night  = next(iter(uploaded_template_night))
template_night  = cv2.imread(template_filename_night)

# Convert both images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_morning_gray = cv2.cvtColor(template_morning, cv2.COLOR_BGR2GRAY)
template_afternoon_gray = cv2.cvtColor(template_afternoon, cv2.COLOR_BGR2GRAY)
template_night_gray = cv2.cvtColor(template_night, cv2.COLOR_BGR2GRAY)

# Enhance contrast using histogram equalization
main_image_eq = cv2.equalizeHist(main_gray)
template_morning_eq = cv2.equalizeHist(template_morning_gray)
template_afternoon_eq = cv2.equalizeHist(template_afternoon_gray)
template_night_eq = cv2.equalizeHist(template_night_gray)

w_morning, h_morning = template_morning_gray.shape[::-1]
w_afternoon, h_afternoon = template_afternoon_gray.shape[::-1]
w_night, h_night = template_night_gray.shape[::-1]
 
res_morning = cv2.matchTemplate(main_image_eq,template_morning_eq,cv2.TM_CCOEFF_NORMED)
res_afternoon = cv2.matchTemplate(main_image_eq,template_afternoon_eq,cv2.TM_CCOEFF_NORMED)
res_night = cv2.matchTemplate(main_image_eq,template_night_eq,cv2.TM_CCOEFF_NORMED)

threshold = 0.68
loc = np.where( res_morning >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_morning, pt[1] + h_morning), (0,0,255), 2)

threshold = 0.6
loc = np.where( res_afternoon >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_afternoon, pt[1] + h_afternoon), (0,255,0), 2)

threshold = 0.56
loc = np.where( res_night >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w_night, pt[1] + h_night), (255,0,0), 2)
    
output_filename = 'res.png'
cv2.imwrite(output_filename,main_image)
files.download(output_filename)