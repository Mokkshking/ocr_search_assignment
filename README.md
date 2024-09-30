# OCR Assignment App
## Overview
The OCR Assignment App is a web-based application designed to perform Optical Character Recognition (OCR) on images containing Hindi and English text. The application leverages two approaches for OCR:

**EasyOCR** for standard OCR functionality, supporting both Hindi and English languages.
**Qwen2VL**, a powerful vision-language model from Huggingface Transformers, for advanced image-to-text extraction and multi-modal capabilities.
Users can upload images, extract text using both techniques, and perform keyword searches on the extracted content, with automatic highlighting of the matched keywords.

![Interface](Streamlit_images/Screenshot (150).png)

**Features**
## Dual OCR Methods:

**EasyOCR:** Quick and lightweight OCR for extracting English and Hindi text.
**Qwen2VL:** Advanced OCR with multi-modal understanding, capable of processing images with context and more complex features.
Keyword Search & Highlighting: Users can input keywords to search within the extracted text, with results automatically highlighted for easy navigation.

Streamlit Interface: A user-friendly UI where users can upload images, view extracted text from both models, search for keywords, and see the highlighted results.

## Key Technologies Used
**Streamlit:** For building the web interface.
**EasyOCR:** For performing standard OCR on images.
**Qwen2VL (Transformers):** Advanced image-to-text extraction model for more complex understanding of the content in images.
**Pillow (PIL):** For image handling and manipulation.
**NumPy:** For array manipulation, required by EasyOCR.
**Regular Expressions (re):** For keyword search and highlighting.
## Installation
Clone the Repository:

git clone https://github.com/Mokkshking/ocr_search_assignment.git
cd ocr_search_assignment
Install Dependencies: Use pip to install the required Python packages.

pip install streamlit easyocr transformers pillow numpy torch
Run the Application: Launch the Streamlit app locally by running:

streamlit run app.py
Application Workflow
Image Upload:

Users upload an image (jpg, jpeg, png) via the file uploader on the Streamlit interface.
Text Extraction (OCR):

EasyOCR: Once an image is uploaded, EasyOCR is used to extract text from both Hindi and English languages.
Qwen2VL: Additionally, the application uses Qwen2VL, a Huggingface vision-language model, to extract text from the image using a more advanced and contextual approach.
Keyword Search:

Users can input a keyword to search within the extracted text from both models.
The app highlights all matches of the keyword (case-insensitive) for easier review.
Text Highlighting:

All occurrences of the searched keyword are highlighted within the extracted text using HTML’s <mark> tag, making it easy to spot in the search results.
Code Explanation
TextExtractor Class (Qwen2VL Integration)
The TextExtractor class is designed to integrate the Qwen2VL model for extracting text from images. This model can generate more detailed and context-aware text by analyzing both the visual and textual aspects of the image.

**Model Initialization:** The Qwen2VL model and processor are loaded from Huggingface's pre-trained model repository (Qwen2-VL-2B-Instruct).
Text Extraction: The image is processed by the model to extract relevant text. This is done by preparing prompts for the model, applying the processor’s chat template, and generating outputs.
Usage:

The extracted text from Qwen2VL can be compared with EasyOCR’s results to see the difference between a basic OCR approach and a more contextually aware vision-language model.
ocr Function (EasyOCR)
### The ocr function utilizes EasyOCR for standard OCR operations:

It initializes an EasyOCR reader for English and Hindi.
The function takes an uploaded image, converts it to a NumPy array, and extracts the text.
Highlight Function
The highlight_text function uses Python’s re library to find and highlight all instances of a user-specified keyword within the extracted text. It’s case-insensitive and uses HTML tags to highlight keywords in the Streamlit display.

## Streamlit UI
The Streamlit interface provides an intuitive layout:

Users can upload images, see the extracted text from both EasyOCR and Qwen2VL, enter a keyword to search, and view highlighted results.
Future Enhancements
**Advanced Comparison:** Add a comparison feature to highlight differences between EasyOCR and Qwen2VL outputs for enhanced understanding of how the models handle complex or contextually rich images.
**Multi-modal Capabilities:** Further explore Qwen2VL's multi-modal capabilities, including adding descriptions, contextual content generation, or translating text.
Expanded Language Support: Extend the language capabilities of both EasyOCR and Qwen2VL for broader OCR use cases.
**Text Analytics:** Add functionality for analyzing the extracted text (e.g., sentiment analysis, entity recognition, etc.).
Conclusion
The OCR Assignment App combines the simplicity and speed of EasyOCR with the advanced multi-modal capabilities of Qwen2VL, offering users a versatile tool for extracting and processing text from images. The application also provides keyword search functionality, making it easier for users to locate specific information within the extracted content.

This app serves as a foundation for more sophisticated OCR systems, leveraging the latest advancements in vision-language models to improve text extraction and understanding from complex image data.
