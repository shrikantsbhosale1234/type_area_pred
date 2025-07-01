import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

# --------------------------
# Load pre-trained models and encoders
# --------------------------
@st.cache_resource
def load_artifacts():
    with open('model_type.pkl', 'rb') as f:
        model_type = pickle.load(f)
    with open('model_area.pkl', 'rb') as f:
        model_area = pickle.load(f)
    with open('model_supply.pkl', 'rb') as f:
        model_supply = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('le_type.pkl', 'rb') as f:
        le_type = pickle.load(f)
    with open('le_area.pkl', 'rb') as f:
        le_area = pickle.load(f)
    with open('le_supply.pkl', 'rb') as f:
        le_supply = pickle.load(f)

    return model_type, model_area, model_supply, tfidf_vectorizer, le_type, le_area, le_supply

# --------------------------
# Prediction function
# --------------------------
def predict_sample(text, model_type, model_area, model_supply, tfidf_vectorizer, le_type, le_area, le_supply):
    if pd.isna(text) or str(text).strip() == "":
        return "Unknown", "Unknown", "Unknown"

    try:
        text_vector = tfidf_vectorizer.transform([str(text)])

        type_encoded = model_type.predict(text_vector)[0]
        area_encoded = model_area.predict(text_vector)[0]
        supply_encoded = model_supply.predict(text_vector)[0]

        try:
            type_label = le_type.inverse_transform([type_encoded])[0]
        except:
            type_label = "Unknown"

        try:
            area_label = le_area.inverse_transform([area_encoded])[0]
        except:
            area_label = "Unknown"

        try:
            supply_label = le_supply.inverse_transform([supply_encoded])[0]
        except:
            supply_label = "Unknown"

        return type_label, area_label, supply_label

    except Exception as e:
        return "Unknown", "Unknown", "Unknown"

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Predict Type/Area/Supply", layout="centered")
st.title("📄 Predict Type, Area, and Supply from Excel")

uploaded_file = st.file_uploader("📤 Upload an Excel file with a 'Text' column", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'Text' not in df.columns:
            st.error("❌ Column 'Text' is required in the uploaded file.")
        else:
            st.success("✅ File uploaded successfully!")

            # Normalize supply column
            if 'Status Supplier 2' in df.columns:
                df['Status Supplier 2'] = np.where(df['Status Supplier 2'] == 'Accepted', 'Accepted', 'Not Accepted')

            # Load models and encoders
            model_type, model_area, model_supply, tfidf_vectorizer, le_type, le_area, le_supply = load_artifacts()

            # Generate predictions
            st.info("🔍 Predicting...")
            type_preds, area_preds, supply_preds = [], [], []

            for text in df['Text']:
                type_out, area_out, supply_out = predict_sample(
                    text,
                    model_type,
                    model_area,
                    model_supply,
                    tfidf_vectorizer,
                    le_type,
                    le_area,
                    le_supply
                )
                type_preds.append(type_out)
                area_preds.append(area_out)
                supply_preds.append(supply_out)

            # Add predictions to the DataFrame
            df['Type_pred'] = type_preds
            df['Area_pred'] = area_preds
            df['Supply_pred'] = supply_preds

            # Display the result
            st.dataframe(df)

            # Prepare Excel for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)

            st.download_button(
                label="📥 Download predictions as Excel",
                data=output,
                file_name="predicted_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
