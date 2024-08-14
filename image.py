from pdf2image import convert_from_path
import pytesseract

# Convert PDF to images
images = convert_from_path('your_file.pdf')

# Apply OCR to each image
for image in images:
    text = pytesseract.image_to_string(image)
    print(text)
