import boto3

import io
from PIL import Image, ImageDraw

# Initialize the AWS Textract client
client = boto3.client('textract')

# Path to your local document/image
document_path = '/Users/eishahemchand/Mineral-Intelligence/courthouse_docs/5325285_83435886_docimage_actual.pdf'  # Update this to the path of your document

# Read the document/image from your local filesystem
with open(document_path, 'rb') as document:
    image_bytes = document.read()

# Use the AWS Textract client to detect text from the bytes of the image
response = client.detect_document_text(Document={'Bytes': image_bytes})

# Get the text blocks from the response
blocks = response['Blocks']

# If you want to process the image (optional)
image = Image.open(document_path)
width, height = image.size
print('Detected Document Text')

# Example for processing the blocks (printing detected text)
for block in blocks:
    if block['BlockType'] == 'LINE':
        print('Detected line:', block['Text'])