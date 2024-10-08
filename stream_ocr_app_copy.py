import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

def extract_text(image_path, processor, model):
    # Open the image
    image = Image.open(image_path)

    # Prepare the messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": "Extract the text from image"
                }
            ]
        }
    ]

    # Create the text prompt using the processor
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare the inputs for the model
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to CPU
    inputs = inputs.to("cpu")

    # Generate output from the model
    output_ids = model.generate(**inputs, max_new_tokens=1024)

    # Extract the generated text
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text

# OCR function using EasyOCR
def ocr(image):
    # Initialize EasyOCR reader for English and Hindi
    reader = easyocr.Reader(['en', 'hi'])

    # Convert PIL image to a NumPy array (EasyOCR expects this format)
    image_np = np.array(image)

    # Perform OCR on the NumPy array
    results = reader.readtext(image_np)

    # Extract and combine the detected text
    extracted_text = " ".join([result[1] for result in results])
    return extracted_text


# Function to highlight keywords in the extracted text
def highlight_text(text, keyword):
    # Use regular expression to find all matches of the keyword (case insensitive)
    highlighted_text = re.sub(f"({keyword})", r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted_text


# Streamlit UI
st.markdown("<h1 style='text-align: center; color: green;'>Ocr Assignment App</h1>", unsafe_allow_html=True)

# Image upload section
uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the image using PIL
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform OCR using EasyOCR
    st.write("Extracting text from the image...")
    extracted_text = ocr(image)
    # For using Qwen2_vl
    extracted_text2 = extract_text(image, processor, model)

    # Display the extracted text
    st.subheader("Extracted Text")
    st.text_area("Extracted Text", extracted_text, height=200)
    st.subheader("Extracted Text from Qwen2")
    st.text_area("Extracted Text", extracted_text2, height=200)

    # Keyword search section
    keyword = st.text_input("Enter keyword to search in the text")

    if keyword:
        # Highlight the keyword in the extracted text
        highlighted_text = highlight_text(extracted_text, keyword)
        highlighted_text2 = highlight_text(extracted_text2, keyword)

        # Display the search results with highlighted keywords
        st.subheader("Search Results")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        st.subheader("Search Results of Qwen2")
        st.markdown(highlighted_text2, unsafe_allow_html=True)
