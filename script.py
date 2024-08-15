# Installing Tesseract OCR
!apt-get install tesseract-ocr
!pip install pytesseract

import cv2
import pytesseract
from google.colab import files
import re
from pytesseract import Output

# Function to clean and normalize a name
def clean_name(name):
    # Remove leading numbers and special characters, and trim spaces
    cleaned = re.sub(r'^[^A-Z]*', '', name).strip()
    return cleaned

# Install Tesseract OCR if not already installed (only necessary if running in an environment like Google Colab)
# !apt-get install tesseract-ocr
# !pip install pytesseract

# Upload an image (use this in Colab or similar platforms)
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert image to grayscale for better OCR performance
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply some pre-processing to improve OCR accuracy
gray_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Use Tesseract to do OCR on the image
details = pytesseract.image_to_data(gray_image, output_type=Output.DICT)

# Extract relevant information and store in a structured format
shift_details = []
is_name = []
names = []
repeat_words= []

# Regular expression to find three or more uppercase letters
name_check_pattern = re.compile(r'([A-Z].*?){3,}')
# Regular expression to check if the string ends with a comma
comma_check_pattern = re.compile(r',\s*$')
# Regular expression to keep only letters and spaces
clean_pattern = re.compile(r'[^\w\s]', re.ASCII)

words_found = 0
last = False
name_words = 3
for text in details['text']:
    text = text.strip()
    if text:  # If text is not empty
        shift_details.append(text)
        # Check if the text matches the pattern for names
        if name_check_pattern.search(text):
            is_name.append(True)
            words_found += 1
        else:
            is_name.append(False)
            repeat_words.append(words_found)
            words_found = 0

        if last:
            names.append(' '.join(shift_details[-words_found:]))
            words_found = 0
        elif  words_found == 3:
            names.append(' '.join(shift_details[-words_found:]))
            words_found = 0

        if comma_check_pattern.search(text):
            last = True
            words_found = 3
        else:
            last = False

# Cleaning names
cleaned_names = [clean_name(name) for name in names]
# Removing duplicates while keeping the most complete name
unique_names = {}
for name in cleaned_names:
    base_name = ' '.join(name.split()[:3])  # Use first three words as a base identifier
    if base_name not in unique_names:
        unique_names[base_name] = name
    else:
        # Keep the longer name if base_name already exists
        if len(name) > len(unique_names[base_name]):
            unique_names[base_name] = name

# Get the cleaned and unique names
cleaned_names = list(unique_names.values())

#cleaned_names = [clean_pattern.sub('', name).strip() for name in names]

# Print the results
print("Shift Details:", shift_details)
print("Is Name:", is_name)
print("Complete Names", names)
print("Cleaned Names", cleaned_names)


