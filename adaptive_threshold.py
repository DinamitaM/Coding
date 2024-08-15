# Install the required libraries
!pip install opencv-python-headless pytesseract pillow

# Download and install Tesseract OCR
!apt-get update
!apt-get install -y tesseract-ocr

import cv2
import pytesseract
from PIL import Image
import numpy as np
from google.colab import files

# Upload your image file
uploaded = files.upload()

# Load the uploaded image
image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply adaptive thresholding
adaptive_threshold_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Save and display the preprocessed image
preprocessed_image_path = 'preprocessed_image.jpg'
cv2.imwrite(preprocessed_image_path, adaptive_threshold_image)

files.download(preprocessed_image_path)