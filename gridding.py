# Install necessary libraries
!pip install pytesseract opencv-python-headless matplotlib

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import Output
from google.colab import files

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

# Function to detect text in each grid cell
def extract_grid_text(image, grid_size=(3, 3)):
    h, w = image.shape
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    
    matrix = []
    
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell_image = image[y1:y2, x1:x2]
            
            # OCR to extract text from cell
            text = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
            text_cheking = text.upper()
            letters_to_check = {'T', 'N', 'M'}
    
            # Check if any of the letters are present in the text
            for letter in letters_to_check:
                text = ' ,'
                if letter in text_cheking:
                    text = letter + ','
                    break;
            
            row.append(text)
        
        matrix.append(row)
    
    return matrix

# Function to upload and process image
def upload_and_process_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        # Load the image
        image = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        # Extract text from grid cells
        grid_size = (47, 31)  # Update with your grid size
        matrix = extract_grid_text(preprocessed_image, grid_size)
        return matrix, image

# Upload and process image
matrix, image = upload_and_process_image()

# Print the result matrix
print("Result Matrix:")
for row in matrix:
    print(' '.join(row))

# Optional: Visualize the grid
def visualize_grid(image, grid_size):
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size[0], w // grid_size[1]
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Grid Visualization')
    plt.axis('off')
    plt.show()

visualize_grid(image, (47, 31))
