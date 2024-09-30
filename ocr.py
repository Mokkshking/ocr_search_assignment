import easyocr
def ocr(image):
# Initialize EasyOCR reader for English and Hindi
    reader = easyocr.Reader(['en', 'hi'])

# Load the image and perform OCR
    image_path = image
    results = reader.readtext(image_path)

# Extract and print the text
    extracted_text = " ".join([result[1] for result in results])
    #print(extracted_text)
    return extracted_text

image = r"C:\Users\Mokksh\Downloads\invoice-test.png"
ocr(image)