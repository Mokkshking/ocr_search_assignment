from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

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


image_path = r"C:\Users\Mokksh\Downloads\invoice-test.png"
extracted_text = extract_text(image_path, processor, model)
print(extracted_text)
