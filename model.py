import pytesseract
from PIL import Image
import cv2
import sys

def load_and_run_ocr(image_path):
    # Load the image using OpenCV
    try:
        img = cv2.imread(image_path)
        # Convert the image to RGB (PyTesseract expects RGB images)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except IOError:
        sys.exit("Unable to open the image file.")

    # # Set the path to the Tesseract executable (edit this according to your installation)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example for Windows
    # # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Example for Unix

    # Use PyTesseract to detect text blocks
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)


    data_points = []

    # Iterate over each text block
    for i in range(len(data['text'])):
        # Check if the block contains text
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            text = data['text'][i]
            data_points.append((x,y))
            print(f"Text Block {i}: '{text}' at position ({x}, {y})")
    return data_points
    

# Example usage
if __name__ == "__main__":
    image_file = '/Users/eishahemchand/Mineral-Intelligence/png docs/2c91ef63-6e85-4225-ac36-20f0416647c0.png'  # Replace with your image file path
    load_and_run_ocr(image_file)
