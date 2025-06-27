import streamlit as st
import pandas as pd
import pickle
import io

# --------------------------
# Prediction Function
# --------------------------
def predict_sample(text, model_type, model_area, tfidf_vectorizer, le_type, le_area):
    # Transform the input text using the fitted TF-IDF vectorizer
    text_vector = tfidf_vectorizer.transform([text])
    
    # Predict encoded labels
    type_pred_encoded = model_type.predict(text_vector)[0]
    area_pred_encoded = model_area.predict(text_vector)[0]
    
    # Decode labels back to original strings
    type_pred_label = le_type.inverse_transform([type_pred_encoded])[0]
    area_pred_label = le_area.inverse_transform([area_pred_encoded])[0]
    
    return type_pred_label, area_pred_label

# --------------------------
# Load Pickle Files
# --------------------------
@st.cache_resource
def load_models():
    with open('model_type.pkl', 'rb') as f:
        model_type = pickle.load(f)
    with open('model_area.pkl', 'rb') as f:
        model_area = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('le_type.pkl', 'rb') as f:
        le_type = pickle.load(f)
    with open('le_area.pkl', 'rb') as f:
        le_area = pickle.load(f)
    return model_type, model_area, tfidf_vectorizer, le_type, le_area

model_type, model_area, tfidf_vectorizer, le_type, le_area = load_models()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Batch Prediction App", layout="centered")
st.title("📁 Predict Type and Area from Uploaded CSV")

uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("❌ The CSV must contain a 'text' column.")
        else:
            st.success("✅ File uploaded successfully!")

            # Apply predictions
            st.info("🔍 Predicting Type and Area...")
            df['Predicted_Type'], df['Predicted_Area'] = zip(*df['text'].apply(
                lambda x: predict_sample(x, model_type, model_area, tfidf_vectorizer, le_type, le_area)
            ))

            st.dataframe(df[['text', 'Predicted_Type', 'Predicted_Area']])

            # Download link
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

