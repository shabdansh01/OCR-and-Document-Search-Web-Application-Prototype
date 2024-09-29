import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import json


@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

def perform_ocr(image):
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Extract the text from the image."},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)


        output_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
        ]
        extracted_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return extracted_text.strip()
    except Exception as e:
        st.error(f"An error occurred during OCR: {e}")
        return ""

st.title("Hindi-English OCR Web Application")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Performing OCR...'):
        extracted_text = perform_ocr(image)
        st.success('OCR Complete!')
    st.subheader("Extracted Text")
    output_format = st.radio("Select output format", ("Plain Text", "JSON"))

    if output_format == "Plain Text":
        st.text_area("Text", extracted_text, height=200)
    elif output_format == "JSON":
        extracted_data = {"extracted_text": extracted_text}
        json_data = json.dumps(extracted_data, ensure_ascii=False, indent=4)
        st.text_area("JSON Output", json_data, height=200)

        # Save extracted text as JSON for download
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="extracted_text.json",
            mime="application/json"
        )


    st.subheader("Keyword Search")
    example_keywords = ["Hindi", "text", "OCR", "example"]
    st.write("Example keywords: ", ", ".join(example_keywords))

    keyword = st.text_input("Enter keyword to search")

    if keyword:
        if keyword in extracted_text:
            st.markdown(f"**Keyword '{keyword}' found in the text.**")
            # Highlight matched sections
            highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            st.markdown(f"**Keyword '{keyword}' not found in the text.**")
