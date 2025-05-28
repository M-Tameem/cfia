import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from transformers import DistilBertTokenizer, TFDistilBertModel
import gc

# --- Page Configuration ---
st.set_page_config(
    page_title="🇨🇦 CFIA Recall Class Predictor",
    page_icon="🇨🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Filepaths ---
BASE_PREPROCESS_PATH = "preprocessed_data_recall_class"
BASE_OUTPUT_PATH = "cfia_analysis_output_v3" # Ensure this dir exists if using this path

TFIDF_PREPROCESSOR_PATH = os.path.join(BASE_PREPROCESS_PATH, "tfidf_preprocessor_rc.pkl")
TABULAR_PREPROCESSOR_NN_PATH = os.path.join(BASE_PREPROCESS_PATH, "tabular_preprocessor_rc.pkl")
RF_MODEL_PATH = "rf_model_recall_class.pkl"
NN_MODEL_H5_PATH = "nn_model_recall_class.h5"
NN_MODEL_KERAS_PATH = "nn_model_recall_class.keras"
NN_MODEL_WEIGHTS_PATH = "nn_model_recall_class.h5"

X_TEST_TFIDF_PATH = os.path.join(BASE_PREPROCESS_PATH, "X_test_tfidf_rc.pkl")
X_TEST_TABULAR_NN_PATH = os.path.join(BASE_PREPROCESS_PATH, "X_test_tabular_rc.pkl")
X_TEST_BERT_EMBEDDINGS_PATH = os.path.join(BASE_PREPROCESS_PATH, "X_test_bert_embeddings_rc.pkl")
Y_TEST_PATH = os.path.join(BASE_PREPROCESS_PATH, "y_test_rc.pkl")
FULL_DATASET_PATH = "cfia_analysis_output_v3/cfia_enhanced_dataset_ml.csv"

BERT_MODEL_NAME = 'distilbert-base-uncased'

# --- Define Model Building Function ---
def build_nn_model(input_dim, num_classes):
    """Defines the NN architecture exactly as in training."""
    model = Sequential([
        Input(shape=(input_dim,), name='input_layer_1'),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Caching Loaded Resources ---
@st.cache_resource
def load_resources():
    """Loads all necessary models, preprocessors, and data."""
    resources = {}
    nn_model_loaded = False

    try:
        # --- Load Full Dataset FIRST to get columns and options ---
        if os.path.exists(FULL_DATASET_PATH):
            resources['full_df'] = pd.read_csv(FULL_DATASET_PATH)
            resources['original_training_columns'] = resources['full_df'].columns.tolist()
            
            # Extract Area of Concern options
            if 'AREA_OF_CONCERN' in resources['full_df'].columns:
                unique_areas = resources['full_df']['AREA_OF_CONCERN'].dropna().unique()
                resources['area_options'] = sorted([str(area).strip() for area in unique_areas])
                if 'unknown' not in resources['area_options']:
                    resources['area_options'].insert(0, 'unknown')
            else:
                st.warning("AREA_OF_CONCERN column not found, using hardcoded list.")
                resources['area_options'] = sorted(['microbiological contamination', 'extraneous material', 'allergen - undeclared',
                                                    'chemical contamination', 'other', 'unknown'] + ['listeria monocytogenes', 'salmonella', 'e. coli o157:h7'])
            resources['primary_options'] = ["Yes", "No", "Unknown"]

        else:
            st.error(f"Missing Essential File: {FULL_DATASET_PATH}. Cannot load.")
            return None

        # Load TFIDF Preprocessor
        with open(TFIDF_PREPROCESSOR_PATH, "rb") as f:
            resources['tfidf_preprocessor'] = pickle.load(f)

        # Load Tabular Preprocessor
        if os.path.exists(TABULAR_PREPROCESSOR_NN_PATH):
            with open(TABULAR_PREPROCESSOR_NN_PATH, "rb") as f:
                loaded_tabular_prep = pickle.load(f)
                resources['tabular_preprocessor_nn'] = None if isinstance(loaded_tabular_prep, str) else loaded_tabular_prep
        else:
            resources['tabular_preprocessor_nn'] = None
            st.sidebar.warning(f"NN Tabular Preprocessor not found: {TABULAR_PREPROCESSOR_NN_PATH}")

        # Load RF Model
        with open(RF_MODEL_PATH, "rb") as f:
            resources['rf_model'] = pickle.load(f)

        # Load Data needed for NN Shape
        X_test_tab, X_test_bert, y_test_data = None, None, None
        if os.path.exists(X_TEST_TABULAR_NN_PATH):
            with open(X_TEST_TABULAR_NN_PATH, "rb") as f: X_test_tab = pickle.load(f)
        if os.path.exists(X_TEST_BERT_EMBEDDINGS_PATH):
             with open(X_TEST_BERT_EMBEDDINGS_PATH, "rb") as f: X_test_bert = pickle.load(f)
        if os.path.exists(Y_TEST_PATH):
            with open(Y_TEST_PATH, "rb") as f: y_test_data = pickle.load(f)

        resources['X_test_tabular_nn'] = X_test_tab
        resources['X_test_bert_embeddings'] = X_test_bert
        resources['y_test'] = y_test_data

        # --- Build and Load NN Model using Weights ---
        resources['nn_model'] = None
        if X_test_tab is not None and X_test_bert is not None and y_test_data is not None:
            if os.path.exists(NN_MODEL_WEIGHTS_PATH):
                try:
                    input_dim = X_test_tab.shape[1] + X_test_bert.shape[1]
                    num_classes = pd.Series(y_test_data).nunique()
                    nn_model_rc = build_nn_model(input_dim, num_classes)
                    nn_model_rc.load_weights(NN_MODEL_WEIGHTS_PATH)
                    resources['nn_model'] = nn_model_rc
                    nn_model_loaded = True
                    st.sidebar.success("NN Model: Built & Weights loaded.")
                except Exception as e_weights:
                    st.sidebar.error(f"Failed to load NN weights from {NN_MODEL_WEIGHTS_PATH}: {e_weights}")
            else:
                st.sidebar.warning(f"NN Weights file not found: {NN_MODEL_WEIGHTS_PATH}. Trying full load.")
        else:
             st.sidebar.warning("NN Model: Cannot build via weights (missing test data). Trying full load.")

        # --- Fallback to Full Model Load ---
        if not nn_model_loaded:
            st.sidebar.info("Attempting full NN model load (.keras first)...")
            if os.path.exists(NN_MODEL_KERAS_PATH):
                try:
                    resources['nn_model'] = tf.keras.models.load_model(NN_MODEL_KERAS_PATH, compile=False)
                    nn_model_loaded = True
                    st.sidebar.success("NN Model: Loaded from .keras format.")
                except Exception as e_keras:
                    st.sidebar.warning(f"Failed to load NN from .keras: {e_keras}. Trying .h5.")
            if not nn_model_loaded and os.path.exists(NN_MODEL_H5_PATH):
                try:
                    resources['nn_model'] = tf.keras.models.load_model(NN_MODEL_H5_PATH, compile=False)
                    nn_model_loaded = True
                    st.sidebar.info("NN Model: Loaded from .h5 format (legacy).")
                except Exception as e_h5:
                    st.sidebar.error(f"Failed to load NN from .h5: {e_h5}")
                    resources['nn_model'] = None
            elif not nn_model_loaded:
                 st.sidebar.error("NN Model: Could not load via weights, .keras, or .h5.")

        # --- Load BERT Tokenizer/Model ---
        resources['bert_tokenizer'] = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
        resources['bert_model'] = TFDistilBertModel.from_pretrained(BERT_MODEL_NAME)
        resources['bert_model'].trainable = False

        # --- Load X_test_tfidf ---
        if os.path.exists(X_TEST_TFIDF_PATH):
            with open(X_TEST_TFIDF_PATH, "rb") as f: resources['X_test_tfidf'] = pickle.load(f)
        else: st.error(f"Missing: {X_TEST_TFIDF_PATH}"); resources['X_test_tfidf'] = None

    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.exception(e)
        return None
    return resources

resources = load_resources()

LABEL_MAP = {0: "Class I (High Risk)", 1: "Class II (Moderate Risk)", 2: "Class III (Low Risk)"}

def generate_bert_embeddings(texts, tokenizer, model, max_len=128):
    if not texts or (len(texts) == 1 and not texts[0]): return np.zeros((len(texts), model.config.hidden_size))
    inputs = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

st.title("🇨🇦 CFIA Food Recall Class Predictor")
st.markdown("Welcome! Predict food recall severity using ML trained on CFIA data.")

# --- Main App Logic ---
if not resources:
     st.error("Application cannot start: Resources failed to load. Please check file paths and console errors.")
else:
    st.sidebar.header("⚙️ Mode & Options")
    app_mode = st.sidebar.radio("Choose Mode:", ["🔮 Predict New Recall", "📊 Explore Test Data"])
    model_choice = st.sidebar.selectbox("Select Prediction Model:",
                                        ["Random Forest", "Neural Network (BERT + Tabular)"],
                                        key="model_selector",
                                        disabled=(not resources.get('nn_model')))
    if not resources.get('nn_model') and model_choice == "Neural Network (BERT + Tabular)":
        st.sidebar.warning("NN model not available, defaulting to Random Forest.")
        model_choice = "Random Forest"

    st.sidebar.markdown("---")
    st.sidebar.info("Educational tool based on CFIA data.")

    #==========================================================================
    # PREDICT NEW RECALL MODE
    #==========================================================================
    if app_mode == "🔮 Predict New Recall":
        st.header("🔮 Predict New Recall Class")

        all_cols = resources['original_training_columns']

        # --- Define options based on CSV header & resources ---
        day_cols = ['Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday', 'Day_nan']
        season_cols = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Season_nan']
        depth_cols = ['Depth_consumer', 'Depth_retail/hri', 'Depth_wholesale', 'Depth_nan']

        depth_options_map = {
            'consumer': 'Depth_consumer', 'retail/hri': 'Depth_retail/hri',
            'wholesale': 'Depth_wholesale', 'unknown': 'Depth_nan'
        }
        depth_display_options = sorted(list(depth_options_map.keys()))
        area_opts = resources['area_options']
        primary_opts = resources['primary_options']
        unknown_area_index = area_opts.index('unknown') if 'unknown' in area_opts else 0
        unknown_primary_index = primary_opts.index('Unknown') if 'Unknown' in primary_opts else 0
        unknown_depth_index = depth_display_options.index('unknown') if 'unknown' in depth_display_options else 0

        with st.form(key="new_recall_form"):
            st.subheader("📝 Essential Information")
            c1, c2 = st.columns(2)
            common_name = c1.text_input("Product Common Name", "e.g., Organic Blueberries", key="common_name")
            recall_date = c2.date_input("Recall Date", datetime.today(), key="recall_date")

            area_of_concern = st.selectbox("Area of Concern", area_opts, index=unknown_area_index, key="area_of_concern")

            c3, c4 = st.columns(2)
            primary_recall = c3.selectbox("Primary Recall?", primary_opts, index=unknown_primary_index, key="primary_recall").lower()
            depth_input_key = c4.selectbox("Recall Depth", options=depth_display_options, index=unknown_depth_index, key="depth")

            with st.expander("➕ Optional / Advanced Details"):
                co1, co2 = st.columns(2)
                days_since_brand = co1.number_input("Days Since Last Brand Recall", 0, value=30, key="days_brand")
                brand_recall_freq = co1.number_input("Brand Recall Frequency", 0, value=1, key="brand_freq")
                incident_brands_involved = co1.number_input("# Brands in Incident", 1, value=1, key="inc_brands")
                incident_num_items = co1.number_input("# Items in Incident", 1, value=1, key="inc_items")
                days_since_prod = co2.number_input("Days Since Last Product Recall", 0, value=90, key="days_prod")
                pathogen_freq = co2.number_input("Pathogen/Concern Frequency", 0, value=1, key="path_freq")
                incident_unique_products = co2.number_input("# Unique Products in Incident", 1, value=1, key="inc_prods")
                incident_duration_days = co2.number_input("Incident Duration (Days)", 0, value=0, key="inc_duration")

            submit_button = st.form_submit_button("🚀 Predict Class")

        if submit_button and common_name:
            input_data = {}

            # Add raw/direct inputs
            input_data['COMMON_NAME'] = common_name
            input_data['AREA_OF_CONCERN'] = area_of_concern # Already lowercase
            input_data['PRIMARY_RECALL'] = primary_recall

            # Add numericals
            input_data['DAYS_SINCE_LAST_BRAND_RECALL'] = days_since_brand
            input_data['DAYS_SINCE_LAST_PROD_RECALL'] = days_since_prod
            input_data['BRAND_RECALL_FREQ'] = brand_recall_freq
            input_data['PATHOGEN_FREQ'] = pathogen_freq
            input_data['Incident_BrandsInvolved'] = incident_brands_involved
            input_data['Incident_UniqueProducts'] = incident_unique_products
            input_data['Incident_NumItems'] = incident_num_items
            input_data['Incident_DurationDays'] = incident_duration_days
            input_data['YEAR'] = recall_date.year
            input_data['MONTH'] = recall_date.month
            input_data['WEEK'] = int(recall_date.isocalendar().week) # Ensure it's int
            input_data['QUARTER'] = (recall_date.month - 1) // 3 + 1

            # Determine Day/Season strings
            day_of_week_str = recall_date.strftime("%A")
            season_str = ("Winter" if recall_date.month in [12, 1, 2] else "Spring" if recall_date.month in [3, 4, 5]
                          else "Summer" if recall_date.month in [6, 7, 8] else "Fall")

            # Add *all* Day_ columns, setting the correct one to 1
            for col in day_cols: input_data[col] = 1 if col == f'Day_{day_of_week_str}' else 0
            if f'Day_{day_of_week_str}' not in day_cols: input_data['Day_nan'] = 1

            # Add *all* Season_ columns, setting the correct one to 1
            for col in season_cols: input_data[col] = 1 if col == f'Season_{season_str}' else 0
            if f'Season_{season_str}' not in season_cols: input_data['Season_nan'] = 1

            # Add *all* Depth_ columns, setting the correct one to 1
            target_depth_col = depth_options_map[depth_input_key]
            for col in depth_cols: input_data[col] = 1 if col == target_depth_col else 0
            if target_depth_col not in depth_cols: input_data['Depth_nan'] = 1

            # Create DataFrame and Reindex/Reorder
            input_df = pd.DataFrame([input_data])
            missing_cols = set(all_cols) - set(input_df.columns)
            for c in missing_cols:
                input_df[c] = 0 # Add any truly missing columns as 0
            input_df = input_df[all_cols] # Ensure exact order and columns

            # Ensure boolean/dummy columns are int/float
            for col in all_cols:
                 if input_df[col].dtype == 'bool' or col.startswith(('Day_','Season_','Depth_')):
                      input_df[col] = input_df[col].astype(float) # Use float for TF compatibility

            st.write("Final DataFrame being sent to model (first 50 columns):")
            st.dataframe(input_df.iloc[:, :50].head())


            st.markdown("---"); st.subheader("🔍 Prediction Results")
            try:
                with st.spinner(f"Predicting with {model_choice}..."):
                    if model_choice == "Random Forest":
                        features_rf = resources['tfidf_preprocessor'].transform(input_df)
                        pred = resources['rf_model'].predict(features_rf)[0]
                        proba = resources['rf_model'].predict_proba(features_rf)[0]
                        classes = resources['rf_model'].classes_
                    else: # Neural Network
                        bert_emb = generate_bert_embeddings(input_df['COMMON_NAME'].tolist(), resources['bert_tokenizer'], resources['bert_model'])
                        tab_feat = resources['tabular_preprocessor_nn'].transform(input_df)
                        comb_feat = np.hstack((tab_feat, bert_emb)).astype(np.float32) # Ensure float32 for NN
                        pred_probs_nn = resources['nn_model'].predict(comb_feat)[0]
                        pred = np.argmax(pred_probs_nn)
                        proba = pred_probs_nn
                        classes = np.arange(len(proba))

                st.success(f"Predicted Recall Class: **{LABEL_MAP.get(pred, 'Unknown')}**")
                prob_df = pd.DataFrame({"Class": [LABEL_MAP.get(c, f"Class {c}") for c in classes], "Probability": proba}
                                       ).sort_values("Probability", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

        elif submit_button and not common_name:
            st.warning("Product Common Name is required.")

    #==========================================================================
    # EXPLORE TEST DATA MODE
    #==========================================================================
    elif app_mode == "📊 Explore Test Data":
        st.header("📊 Explore Test Data & Model Performance")
        if not all(resources.get(k) is not None for k in ['y_test', 'full_df', 'X_test_tfidf', 'X_test_tabular_nn', 'X_test_bert_embeddings']):
            st.warning("Test data, labels, or full dataset not fully loaded. Exploration might be limited.")
        else:
            idx = st.slider("Test Sample Index:", 0, len(resources['y_test']) - 1, 0, key="test_slider")
            true_label = resources['y_test'].iloc[idx] if hasattr(resources['y_test'], 'iloc') else resources['y_test'][idx]
            st.subheader(f"🎯 True Class: {LABEL_MAP.get(true_label, 'Unknown')}")

            try:
                orig_idx = resources['y_test'].index[idx] if hasattr(resources['y_test'], 'index') else idx
                st.caption("Original Features (from cfia_enhanced_dataset_ml.csv):")
                st.dataframe(resources['full_df'].iloc[[orig_idx]])
            except Exception as e: st.caption(f"Could not map to original full_df row reliably: {e}")
            st.markdown("---")

            if model_choice == "Random Forest":
                sample = resources['X_test_tfidf'][idx].reshape(1,-1)
                pred = resources['rf_model'].predict(sample)[0]
                st.markdown(f"🤖 **RF Prediction:** **{LABEL_MAP.get(pred, 'Unknown')}** {'✅ Correct' if pred == true_label else '❌ Incorrect'}")
            else: # Neural Network
                tab_sample = resources['X_test_tabular_nn'][idx].reshape(1,-1)
                bert_sample = resources['X_test_bert_embeddings'][idx].reshape(1,-1)
                comb_sample = np.hstack((tab_sample, bert_sample))
                pred_nn = np.argmax(resources['nn_model'].predict(comb_sample)[0])
                st.markdown(f"🧠 **NN Prediction:** **{LABEL_MAP.get(pred_nn, 'Unknown')}** {'✅ Correct' if pred_nn == true_label else '❌ Incorrect'}")

            if st.button("Show Confusion Matrix (Full Test Set)", key="show_cm_btn"):
                fig, ax = plt.subplots()
                y_true = resources['y_test']
                try:
                    if model_choice == "Random Forest":
                        y_pred = resources['rf_model'].predict(resources['X_test_tfidf'])
                        ax.set_title("Random Forest Confusion Matrix")
                    else: # Neural Network
                        tab_all = resources['X_test_tabular_nn']
                        bert_all = resources['X_test_bert_embeddings']
                        comb_all = np.hstack((tab_all, bert_all))
                        y_pred = np.argmax(resources['nn_model'].predict(comb_all), axis=1)
                        ax.set_title("Neural Network Confusion Matrix")

                    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys()))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_MAP.values(), yticklabels=LABEL_MAP.values(), ax=ax)
                    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                    st.pyplot(fig)
                except Exception as e: st.error(f"CM Error: {e}")
                finally: plt.close(fig); gc.collect()