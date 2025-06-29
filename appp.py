import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# Load trained models
# --------------------------
@st.cache_resource
def load_models():
    with open('model_type.pkl', 'rb') as f:
        model_type = pickle.load(f)
    with open('model_area.pkl', 'rb') as f:
        model_area = pickle.load(f)
    return model_type, model_area

# --------------------------
# Encode text and labels
# --------------------------
def encode_text_data(df, text_column='Text', type_column='Type.1', area_column='AREA', max_features=5000):
    le_type = LabelEncoder()
    y_type = le_type.fit_transform(df[type_column])

    le_area = LabelEncoder()
    y_area = le_area.fit_transform(df[area_column])

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df[text_column])

    return X, y_type, y_area, tfidf_vectorizer, le_type, le_area

# --------------------------
# Predict single sample
# --------------------------
def predict_sample(text, model_type, model_area, tfidf_vectorizer, le_type, le_area):
    text_vector = tfidf_vectorizer.transform([str(text)])
    type_encoded = model_type.predict(text_vector)[0]
    area_encoded = model_area.predict(text_vector)[0]
    type_label = le_type.inverse_transform([type_encoded])[0]
    area_label = le_area.inverse_transform([area_encoded])[0]
    return type_label, area_label

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Text to Type/Area Predictor", layout="centered")
st.title("📄 Predict Type and Area from Excel (.xlsx)")

uploaded_file = st.file_uploader("Upload an Excel file with columns: Text, Type.1, AREA", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df=df[['ID','Text','Type.1','AREA']]
        df=df.dropna(subset=['ID','Text','Type.1','AREA'],ignore_index=True)

        # Validate columns
        if not all(col in df.columns for col in ['Text', 'Type.1', 'AREA']):
            st.error("❌ Columns 'Text', 'Type.1', and 'AREA' are required in the Excel file.")
        else:
            st.success("✅ File uploaded successfully!")

            # Load models
            model_type, model_area = load_models()

            # Fit encoders and vectorizer from current data
            X, y_type, y_area, tfidf_vectorizer, le_type, le_area = encode_text_data(df)

            # Make predictions
            st.info("🔍 Predicting...")
            type_preds, area_preds = [], []

            type_preds, area_preds , id = [], [], []
            i=-1
            for text in df['Text']:
                i+=1
                type_out, area_out = predict_sample(text, model_type, model_area, tfidf_vectorizer, le_type,le_area)
                type_preds.append(type_out)
                area_preds.append(area_out)
                id.append(df['ID'][i])
                
            df_preds = df.copy()
            df_preds['Type_pred'] = type_preds
            df_preds['Area_pred'] = area_preds
            df_preds['id'] = id

            # Show preview
            st.dataframe(df_preds)

            # Prepare download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)

            st.download_button(
                label="📥 Download predictions as Excel",
                data=output,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
