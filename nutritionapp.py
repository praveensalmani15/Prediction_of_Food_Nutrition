import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
 
# Load model
model = tf.keras.models.load_model("food_classifier_mobilenet.h5")
 
# Class labels
class_labels = ['biryani', 'burger', 'dosa', 'fries', 'idli', 'pasta', 'pizza', 'salad', 'sandwich', 'upma']
 
# Load nutrition dataset
nutrition_df = pd.read_csv("sample_food_nutrition.csv")
 
# Resize and preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
 
# Predict food class
def predict_food(image, model):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class
 
# Get nutrition info from CSV
def get_nutrition_info(food_label, nutrition_df):
    row = nutrition_df[nutrition_df['food_name'].str.lower() == food_label.lower()]
    if not row.empty:
        row = row.iloc[0]
        return {
            "Calories": row['calories'],
            "Protein": row['protein'],
            "Fat": row['fat'],
            "Carbs": row['carbs']
        }
    else:
        return {
            "Calories": "N/A",
            "Protein": "N/A",
            "Fat": "N/A",
            "Carbs": "N/A"
        }
 
# Streamlit App
st.title("🍔 Food Recognition & Nutrition Estimator")
st.markdown("Upload a food image and get its estimated nutritional values.")
 
uploaded_file = st.file_uploader("📷 Upload Food Image", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
 
    with st.spinner("Analyzing..."):
        food_name = predict_food(image, model)
        nutrition = get_nutrition_info(food_name, nutrition_df)
 
    st.success(f"🍽️ Predicted Food: **{food_name.title()}**")

    st.subheader("🔬 Estimated Nutrition:")
    st.write(f"🔥 **Calories:** {nutrition['Calories']} kcal")
    st.write(f"💪 **Protein:** {nutrition['Protein']} g")
    st.write(f"🥑 **Fat:** {nutrition['Fat']} g")
    st.write(f"🍞 **Carbs:** {nutrition['Carbs']} g")
