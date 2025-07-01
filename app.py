import streamlit as st
import pandas as pd
import pickle
import io

# --------------------------
# Prediction Function
# --------------------------
def predict_sample(text, model_type, model_area, tfidf_vectorizer, le_type, le_area):
    # Transform the input text using the fitted TF-IDF vectorizer
    text_vector = tfidf_vectorizer.transform([str(text)])
    
    # Predict encoded labels
    type_pred_encoded = model_type.predict(text_vector)[0]
    area_pred_encoded = model_area.predict(text_vector)[0]
    
    # Decode labels back to original strings
    type_pred_label = le_type.inverse_transform([type_pred_encoded])[0]
    area_pred_label = le_area.inverse_transform([area_pred_encoded])[0]
    
    return type_pred_label, area_pred_label

# --------------------------
# Load Pickle Models
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
st.set_page_config(page_title="Excel Type/Area Predictor", layout="centered")
st.title("📊 Predict Type and Area from Excel (.xlsx)")

uploaded_file = st.file_uploader("Upload an Excel file with a 'text' column", type=['xlsx'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'text' not in df.columns:
            st.error("❌ The Excel file must contain a 'text' column.")
        else:
            st.success("✅ File uploaded and read successfully!")

            st.info("🔍 Generating predictions...")

            # Make predictions
            df['Predicted_Type'], df['Predicted_Area'] = zip(*df['text'].apply(
                lambda x: predict_sample(x, model_type, model_area, tfidf_vectorizer, le_type, le_area)
            ))

            # Show preview
            st.dataframe(df[['text', 'Predicted_Type', 'Predicted_Area']])

            # Prepare for Excel download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)

            # Download button
            st.download_button(
                label="📥 Download predictions as Excel",
                data=output,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
