import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# Step 1: Load your data
df = pd.read_excel("clear_data.xlsx")  # Replace with your file path
max_features=5000
tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

# Step 2: Encode text and labels
def encode_text_data(df, text_column='Text', type_column='Type.1', area_column='AREA', max_features=5000):
    label_encoder_type = LabelEncoder()
    y_type = label_encoder_type.fit_transform(df[type_column])

    label_encoder_area = LabelEncoder()
    y_area = label_encoder_area.fit_transform(df[area_column])

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df[text_column])

    return X, y_type, y_area, tfidf_vectorizer, label_encoder_type, label_encoder_area

# Encode the data
X, y_type, y_area, tfidf_vectorizer, le_type, le_area = encode_text_data(df)

# Step 3: Train-test split
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X, y_type, test_size=0.2, random_state=42)
X_train_area, X_test_area, y_train_area, y_test_area = train_test_split(X, y_area, test_size=0.2, random_state=42)

# Step 4: Train Random Forest models
model_type = RandomForestClassifier(n_estimators=200, random_state=42)
model_area = RandomForestClassifier(n_estimators=200, random_state=42)

model_type.fit(X_train_type, y_train_type)
model_area.fit(X_train_area, y_train_area)

# Step 5: Make predictions
y_pred_type = model_type.predict(X_test_type)
y_pred_area = model_area.predict(X_test_area)

# Step 6: Evaluation with dynamic label correction
print("=== Type.1 Prediction Results ===")
print("Accuracy:", accuracy_score(y_test_type, y_pred_type))

labels_type = unique_labels(y_test_type, y_pred_type)
target_names_type = le_type.inverse_transform(labels_type)
print(classification_report(y_test_type, y_pred_type, labels=labels_type, target_names=target_names_type))

print("\n=== AREA Prediction Results ===")
print("Accuracy:", accuracy_score(y_test_area, y_pred_area))

labels_area = unique_labels(y_test_area, y_pred_area)
target_names_area = le_area.inverse_transform(labels_area)
print(classification_report(y_test_area, y_pred_area, labels=labels_area, target_names=target_names_area))




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
# Load Models and Encoders
# --------------------------
@st.cache_resource
def load_all_models():
    with open('model_type.pkl', 'rb') as f:
        model_type = pickle.load(f)
    with open('model_area.pkl', 'rb') as f:
        model_area = pickle.load(f)
    
    return model_type, model_area, tfidf_vectorizer, le_type, le_area

# Load everything once
model_type, model_area, tfidf_vectorizer, le_type, le_area = load_all_models()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Text Classifier", layout="centered")

st.title("🔍 Predict Type and Area from Text")

text_input = st.text_area("Enter your input text below:", height=150)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        type_label, area_label = predict_sample(
            text_input, model_type, model_area, tfidf_vectorizer, le_type, le_area
        )
        
        st.success(f"✅ **Predicted Type:** `{type_label}`")
        st.success(f"✅ **Predicted Area:** `{area_label}`")
