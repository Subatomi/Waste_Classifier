import streamlit as st
import tensorflow as tf
import google.generativeai as genai
from PIL import Image
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import numpy as np

# Load tokenizer and model
generate_model = AutoModelForCausalLM.from_pretrained("./gpt2-waste-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-waste-finetuned")

# Generate a suggestion based on material
def generate_suggestion(material):
    prompt = f"Material: {material}\nSuggestion:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = generate_model.generate(
        **inputs,
        max_length=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggestion.replace(prompt, "").strip()


# Configure Gemini API
genai.configure(api_key="AIzaSyCHQNz9gKTXaSloeKlj6PZ4-AcI4cjONVs")  # Replace with your actual API key

# Load custom model
model = tf.keras.models.load_model("recycling_model.h5")

# Define class labels
label_names = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction
def predict_material(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = label_names[predicted_class]
    confidence = predictions[0][predicted_class]
    return predicted_label, confidence

# Gemini suggestion
def get_material_suggestion(predicted_label):
    prompt = f"""
    I have an item made of {predicted_label}. Please provide a 3 top steps on how or friendly suggestion on how to properly dispose of or reuse this material.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Dummy custom model suggestion (for demo)
def custom_model_suggestion(predicted_label):
    return f"This is a static suggestion for handling {predicted_label}. (Replace with your own logic.)"

# Streamlit UI
st.set_page_config(layout="wide")
page_bg_img = f"""
<style>
[data-testid="stColumn"] {{
background-color: white;
padding: 3em;
color: black;
border-radius: 10px;
border: 2px solid #6792e0;
box-shadow: 4px 4px #6792e0;
}}

[data-testid="stMarkdownContainer"]{{
color: black;
}}

#recycling-material-classifier-and-suggestion{{
color: white;
}}

.stButton.st-emotion-cache-8atqhb.e1mlolmg0 button{{
  background-color: #6792e0; /* A nice green */
  color: white;
  padding: 12px 24px;
  border: 1px solid #000;
  box-shadow: 4px 4px #000;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s ease; /* Smooth transition for hover */
}}

.stButton.st-emotion-cache-8atqhb.e1mlolmg0 button:hover {{
  background-color: #326dd9;
  color: white;
}}

 .st-emotion-cache-70qvj9.e14dwhi25{{
 color: white;
}}



</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("♻️ Recycling Material Classifier and Suggestion")

# Layout: Three Columns
left_col, center_col, right_col = st.columns([1, 1.5, 1])

# File uploader in left column
with left_col:
    st.header("Input Image")

    input_method = st.radio("Choose input method:", ["Upload", "Camera"], index=0)

    if input_method == "Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a photo")


# Center column: material header and image
with center_col:
    predicted_label = None
    st.header("Material: (Put Material Name of Classified image)")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        classify_clicked = st.button("Classify")
    else:
        st.markdown("🖼️ *No image uploaded yet.*")
        classify_clicked = False

# Right column: Suggestion method + output
with right_col:
    st.header("Suggestion")

    suggestion_model = st.radio(
        "Choose suggestion model:",
        ["Gemini 2.0", "Custom Model"],
        index=0
    )

    if uploaded_file is not None and classify_clicked:
        predicted_label, confidence = predict_material(image)

        # Update center column header with predicted label
        with center_col:
            st.header(f"Material: {predicted_label.capitalize()}")

        # Get suggestion based on selected model
        if suggestion_model == "Gemini 2.0":
            suggestion = get_material_suggestion(predicted_label)
        else:
            suggestion = generate_suggestion(predicted_label)

        st.markdown(f"**Predicted Material:** {predicted_label} ({confidence * 100:.2f}%)")
        st.markdown("**Suggested Action:**")
        st.write(suggestion)
    else:
        st.markdown("💡 *No prediction yet. Suggestions will appear here.*")
