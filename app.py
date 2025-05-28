import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from collections import defaultdict, Counter
import graphviz # For graphtime.py visualization
import math
import pickle # For newfrontend.py
from datetime import datetime # For newfrontend.py
from sklearn.metrics import confusion_matrix # For newfrontend.py
import tensorflow as tf # For newfrontend.py
from tensorflow.keras.models import Sequential # For newfrontend.py
from tensorflow.keras.layers import Dense, Dropout, Input # For newfrontend.py
from transformers import DistilBertTokenizer, TFDistilBertModel # For newfrontend.py
import gc # For newfrontend.py
import matplotlib.pyplot as plt


# --- Environment Variable (from newfrontend.py) ---
# This should be one of the first things set, before TensorFlow might be imported elsewhere implicitly
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# --- Page Configuration (Call this once at the beginning) ---
st.set_page_config(
    page_title="CFIA Data Analysis Suite",
    page_icon="🇨🇦", # Using one of the icons, can be changed
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants and Helper Functions for Brand Association Analyzer (from graphtime.py) ---

# Configuration for graphtime
GT_DEFAULT_DATA_PATH = './cfia_analysis_output_v3/'
GT_CONNECTIONS_FILE = 'cfia_brand_connections.csv'
GT_ML_DATA_FILE = 'cfia_enhanced_dataset_ml.csv'
GT_PRODUCT_SIMILARITY_THRESHOLD = 0.3
GT_TOP_N_DEFAULT = 10
GT_NUM_EXAMPLES_TO_SHOW = 5
GT_NUM_CLUSTERS = 6

# Ranking Weights for graphtime
GT_W_CONN = 0.50
GT_W_PROD_SIM = 0.30
GT_W_CONT_MATCH = 0.15
GT_W_PRIMARY_RECALL = 0.05

@st.cache_resource
def gt_load_brand_graph(connections_path):
    if not os.path.exists(connections_path): return None, 0
    try:
        connections_df = pd.read_csv(connections_path)
        G = nx.Graph()
        max_weight = 0
        for _, row in connections_df.iterrows():
            brand1 = str(row['Brand1']).lower().strip()
            brand2 = str(row['Brand2']).lower().strip()
            weight = row['Weight']
            if brand1 and brand2:
                 G.add_edge(brand1, brand2, weight=weight)
                 if weight > max_weight: max_weight = weight
        return G, max_weight if max_weight > 0 else 1
    except Exception as e:
        st.error(f"Error loading brand connections graph: {e}")
        return None, 0

@st.cache_data
def gt_load_ml_data(ml_data_path):
    if not os.path.exists(ml_data_path): return None
    try:
        df = pd.read_csv(ml_data_path, parse_dates=['RECALL_DATE'], infer_datetime_format=True)
        df['BRAND_NAME'] = df['BRAND_NAME'].astype(str).str.lower().str.strip()
        df['COMMON_NAME'] = df['COMMON_NAME'].astype(str).str.lower().str.strip()
        df['AREA_OF_CONCERN'] = df['AREA_OF_CONCERN'].astype(str).str.lower().str.strip()
        if 'PRIMARY_RECALL' in df.columns:
            df['PRIMARY_RECALL'] = df['PRIMARY_RECALL'].astype(str).str.lower().str.strip()
        if 'RECALL_DATE' in df.columns:
            df['RECALL_DATE'] = pd.to_datetime(df['RECALL_DATE'], errors='coerce')
        for col in ['Incident_NumItems', 'Incident_BrandsInvolved', 'Incident_UniqueProducts', 'Incident_DurationDays']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading ML dataset for Brand Analyzer: {e}")
        return None

@st.cache_resource
def gt_get_tfidf_vectorizer_and_matrix(_ml_df):
    if _ml_df is None: return None, None, []
    product_names = _ml_df['COMMON_NAME'].unique().tolist()
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(product_names)
    return vectorizer, tfidf_matrix, product_names

def gt_get_product_similarity(query_product, vectorizer, product_tfidf_matrix, all_product_names):
    if not query_product or vectorizer is None: return {}
    query_vector = vectorizer.transform([query_product.lower().strip()])
    cosine_similarities = cosine_similarity(query_vector, product_tfidf_matrix).flatten()
    return {name: score for name, score in zip(all_product_names, cosine_similarities)}

@st.cache_data
def gt_calculate_brand_profiles(_ml_df):
    if _ml_df is None: return None
    profiles = {}
    brand_groups = _ml_df.groupby('BRAND_NAME')
    for brand, group in brand_groups:
        if not brand or brand == 'unknown': continue
        class_counts = group['Incident_Severity'].value_counts(normalize=True) * 100
        contaminant_counts = group['AREA_OF_CONCERN'].value_counts()
        profiles[brand] = {
            'Total Recalls': len(group.drop_duplicates(subset=['RECALL_NUMBER', 'COMMON_NAME'])),
            'Unique Incidents': group['RECALL_NUMBER'].nunique(),
            'Class I Pct': class_counts.get('Class I', 0.0),
            'Class II Pct': class_counts.get('Class II', 0.0),
            'Class III Pct': class_counts.get('Class III', 0.0),
            'Unique Products': group['COMMON_NAME'].nunique(),
            'Avg Incident Size': group['Incident_NumItems'].mean(),
            'Avg Brands Involved': group['Incident_BrandsInvolved'].mean(),
            'Multi-Brand Rate': (group['Incident_BrandsInvolved'] > 1).mean() * 100,
            'Top Contaminant': contaminant_counts.idxmax() if not contaminant_counts.empty else 'N/A',
            'Top Contaminant Count': contaminant_counts.max() if not contaminant_counts.empty else 0
        }
    return pd.DataFrame.from_dict(profiles, orient='index').fillna(0)

@st.cache_data
def gt_get_brand_clusters(_profiles_df, n_clusters=GT_NUM_CLUSTERS):
    if _profiles_df is None or _profiles_df.empty: return None
    features = [
        'Total Recalls', 'Unique Incidents', 'Class I Pct', 'Class II Pct',
        'Unique Products', 'Avg Incident Size', 'Avg Brands Involved', 'Multi-Brand Rate'
    ]
    features_to_use = [f for f in features if f in _profiles_df.columns]
    if not features_to_use: return None
    X = _profiles_df[features_to_use].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init explicitly
    cluster_labels = kmeans.fit_predict(X_scaled)
    _profiles_df_clustered = _profiles_df.copy()
    _profiles_df_clustered['Cluster'] = cluster_labels
    return _profiles_df_clustered

@st.cache_data
def gt_find_indirect_connections(_brand_graph, selected_brand, top_n=10):
    if not selected_brand or selected_brand not in _brand_graph:
        return pd.DataFrame()
    indirect = defaultdict(float)
    direct_neighbors = set(_brand_graph.neighbors(selected_brand)) if selected_brand in _brand_graph else set()
    for neighbor in direct_neighbors:
        weight1 = _brand_graph[selected_brand][neighbor].get('weight', 0)
        for second_neighbor in _brand_graph.neighbors(neighbor):
            if second_neighbor != selected_brand and second_neighbor not in direct_neighbors:
                weight2 = _brand_graph[neighbor][second_neighbor].get('weight', 0)
                indirect[second_neighbor] += weight1 * weight2
    if not indirect: return pd.DataFrame()
    indirect_df = pd.DataFrame(list(indirect.items()), columns=['Indirect Brand', 'Connection Score'])
    return indirect_df.sort_values('Connection Score', ascending=False).head(top_n).reset_index(drop=True)

def gt_calculate_suggestion_score(suggestion_details, input_contaminant_query, max_connection_weight):
    score = 0
    norm_conn_weight = (suggestion_details.get("Connection Weight", 0) / max_connection_weight) if max_connection_weight else 0
    score += norm_conn_weight * GT_W_CONN
    prod_sim = suggestion_details.get("Product Similarity", 0.0)
    if not isinstance(prod_sim, (float, int)): prod_sim = 0.0
    score += prod_sim * GT_W_PROD_SIM
    if input_contaminant_query and suggestion_details.get("Original Contaminant", "").lower() == input_contaminant_query.lower():
        score += 1.0 * GT_W_CONT_MATCH
    if 'PRIMARY_RECALL' in suggestion_details and str(suggestion_details.get("PRIMARY_RECALL", "")).lower() == 'yes':
        score += 1.0 * GT_W_PRIMARY_RECALL
    return round(score, 4)

def gt_generate_graph_viz(brand_graph, selected_brand):
    if not selected_brand or selected_brand not in brand_graph: return None
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    dot.node(selected_brand.title(), shape='ellipse', style='filled', color='skyblue')
    max_neighbors_to_show = 15
    neighbors_count = 0
    # Ensure neighbors are sorted or handled consistently if graph is large
    # For simplicity, taking them as they come from brand_graph.neighbors
    sorted_neighbors = sorted(list(brand_graph.neighbors(selected_brand)), key=lambda n: brand_graph[selected_brand][n].get('weight', 0), reverse=True)

    for neighbor in sorted_neighbors:
        if neighbors_count >= max_neighbors_to_show:
            dot.node("...", label="...", shape='plaintext'); break
        weight = brand_graph[selected_brand][neighbor].get('weight', 0)
        dot.node(neighbor.title(), shape='ellipse')
        dot.edge(selected_brand.title(), neighbor.title(), label=f'w: {weight:.2f}') # Formatting weight
        neighbors_count +=1
    return dot if neighbors_count > 0 else None


def gt_run_association_analysis(selected_brand, input_product_query, input_contaminant_query,
                             brand_graph, max_graph_weight, ml_df,
                             tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf,
                             product_similarity_threshold_input, top_n_suggestions):
    if not selected_brand or ml_df is None or brand_graph is None: return pd.DataFrame() # Added checks for ml_df and brand_graph
    associated_brands_scores = []
    if selected_brand in brand_graph:
        for neighbor in brand_graph.neighbors(selected_brand):
            weight = brand_graph[selected_brand][neighbor].get('weight', 0)
            associated_brands_scores.append({'brand': neighbor, 'connection_weight': weight})
    associated_brands_scores = sorted(associated_brands_scores, key=lambda x: x['connection_weight'], reverse=True)
    suggestions_list, processed_recalls = [], set()
    for assoc_info in associated_brands_scores:
        assoc_brand = assoc_info['brand']; connection_weight = assoc_info['connection_weight']
        brand_recalls_df = ml_df[ml_df['BRAND_NAME'] == assoc_brand]
        if brand_recalls_df.empty: continue
        if input_contaminant_query: # Ensure contaminant query is lowercased for comparison
            brand_recalls_df = brand_recalls_df[brand_recalls_df['AREA_OF_CONCERN'] == input_contaminant_query.lower()]
        if brand_recalls_df.empty: continue
        
        product_similarities_for_brand = {}
        if input_product_query and tfidf_vectorizer and product_tfidf_matrix is not None and all_product_names_for_tfidf:
             product_similarities_for_brand = gt_get_product_similarity(input_product_query, tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf)

        for _, recall_row in brand_recalls_df.iterrows():
            recalled_product_name = recall_row['COMMON_NAME']; similarity_score = 0.0
            passes_product_filter = not input_product_query # True if no product query
            
            if input_product_query: # Only calculate similarity if there's a query
                if tfidf_vectorizer and product_similarities_for_brand: # Check if resources are available
                    similarity_score = product_similarities_for_brand.get(recalled_product_name, 0.0)
                    if similarity_score >= product_similarity_threshold_input: passes_product_filter = True
                else: # If no vectorizer, cannot filter by product similarity effectively
                    passes_product_filter = False # Or handle as per desired logic: maybe match exact name?

            if passes_product_filter:
                recall_key = (assoc_brand, recalled_product_name, recall_row['AREA_OF_CONCERN'], recall_row.get('RECALL_NUMBER', str(_))) # Ensure RECALL_NUMBER is hashable
                if recall_key not in processed_recalls:
                    suggestion_details = {
                        "Associated Brand": assoc_brand.title(), "Recalled Product": recalled_product_name.title(),
                        "Original Contaminant": recall_row['AREA_OF_CONCERN'].title(), "Recall Class": recall_row.get('Incident_Severity', 'N/A'),
                        "Recall Date": pd.to_datetime(recall_row.get('RECALL_DATE')).strftime('%Y-%m-%d') if pd.notna(recall_row.get('RECALL_DATE')) else 'N/A',
                        "Recall Number": recall_row.get('RECALL_NUMBER', 'N/A'), "Connection Weight": connection_weight,
                        "Product Similarity": round(similarity_score, 3) if input_product_query else "N/A", 
                        "PRIMARY_RECALL": recall_row.get('PRIMARY_RECALL', 'N/A')}
                    suggestion_details["Score"] = gt_calculate_suggestion_score(suggestion_details, input_contaminant_query, max_graph_weight)
                    suggestions_list.append(suggestion_details)
                    processed_recalls.add(recall_key)
    sorted_suggestions = sorted(suggestions_list, key=lambda x: x['Score'], reverse=True)
    return pd.DataFrame(sorted_suggestions[:top_n_suggestions]) if sorted_suggestions else pd.DataFrame()

@st.cache_data
def gt_get_example_incidents(_ml_df, num_examples=GT_NUM_EXAMPLES_TO_SHOW):
    if _ml_df is None: return []
    # Ensure RECALL_NUMBER is treated as string if it's mixed type or for grouping
    _ml_df['RECALL_NUMBER_STR'] = _ml_df['RECALL_NUMBER'].astype(str)
    incident_groups = _ml_df.groupby('RECALL_NUMBER_STR')
    multi_brand_incidents = []
    for recall_num_str, group in incident_groups:
        unique_brands = group['BRAND_NAME'].unique()
        if len(unique_brands) > 1:
            primary_brand = unique_brands[0]; other_brands = unique_brands[1:]
            example_row = group.iloc[0]
            multi_brand_incidents.append({
                "recall_number": recall_num_str, "primary_brand_example": primary_brand, "other_brands_involved": list(other_brands),
                "example_product": example_row['COMMON_NAME'], "example_contaminant": example_row['AREA_OF_CONCERN'],
                "incident_severity": example_row.get('Incident_Severity', 'N/A'),
                "recall_date": pd.to_datetime(example_row.get('RECALL_DATE')).strftime('%Y-%m-%d') if pd.notna(example_row.get('RECALL_DATE')) else 'N/A',
                "full_incident_details_df": group[['BRAND_NAME', 'COMMON_NAME', 'AREA_OF_CONCERN', 'Incident_Severity', 'RECALL_DATE']].drop_duplicates().reset_index(drop=True)})
            if len(multi_brand_incidents) >= num_examples: break
    return multi_brand_incidents

# --- Constants and Helper Functions for Recall Class Predictor (from newfrontend.py) ---

# Filepaths for newfrontend
NF_BASE_PREPROCESS_PATH = "preprocessed_data_recall_class"
NF_BASE_OUTPUT_PATH = "cfia_analysis_output_v3"
NF_TFIDF_PREPROCESSOR_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "tfidf_preprocessor_rc.pkl")
NF_TABULAR_PREPROCESSOR_NN_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "tabular_preprocessor_rc.pkl")
NF_RF_MODEL_PATH = "rf_model_recall_class.pkl"
NF_NN_MODEL_H5_PATH = "nn_model_recall_class.h5" # Used as weights and full model
NF_NN_MODEL_KERAS_PATH = "nn_model_recall_class.keras" # Preferred full model format
NF_X_TEST_TFIDF_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "X_test_tfidf_rc.pkl")
NF_X_TEST_TABULAR_NN_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "X_test_tabular_rc.pkl")
NF_X_TEST_BERT_EMBEDDINGS_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "X_test_bert_embeddings_rc.pkl")
NF_Y_TEST_PATH = os.path.join(NF_BASE_PREPROCESS_PATH, "y_test_rc.pkl")
NF_FULL_DATASET_PATH = os.path.join(NF_BASE_OUTPUT_PATH, "cfia_enhanced_dataset_ml.csv") # Note: This is same as GT_ML_DATA_FILE path

NF_BERT_MODEL_NAME = 'distilbert-base-uncased'
NF_LABEL_MAP = {0: "Class I (High Risk)", 1: "Class II (Moderate Risk)", 2: "Class III (Low Risk)"}

def nf_build_nn_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,), name='input_layer_1'), # Explicit Input layer
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

@st.cache_resource
def nf_load_resources():
    resources = {'nn_model_loaded_successfully': False} # Flag to track NN model status
    nn_model_loaded_flag = False # Local flag for loading logic

    try:
        if os.path.exists(NF_FULL_DATASET_PATH):
            resources['full_df'] = pd.read_csv(NF_FULL_DATASET_PATH)
            resources['original_training_columns'] = resources['full_df'].columns.tolist()
            if 'AREA_OF_CONCERN' in resources['full_df'].columns:
                unique_areas = resources['full_df']['AREA_OF_CONCERN'].dropna().unique()
                resources['area_options'] = sorted([str(area).strip().lower() for area in unique_areas]) # ensure lower
                if 'unknown' not in resources['area_options']:
                    resources['area_options'].insert(0, 'unknown')
            else:
                st.sidebar.warning("AREA_OF_CONCERN column not found in full_df, using hardcoded list for Recall Predictor.")
                resources['area_options'] = sorted(['microbiological contamination', 'extraneous material', 'allergen - undeclared',
                                                    'chemical contamination', 'other', 'unknown', 'listeria monocytogenes', 'salmonella', 'e. coli o157:h7']) # ensure lower
            resources['primary_options'] = ["Yes", "No", "Unknown"]
        else:
            st.error(f"Missing Essential File for Recall Predictor: {NF_FULL_DATASET_PATH}.")
            return resources # Return partially filled resources

        if os.path.exists(NF_TFIDF_PREPROCESSOR_PATH):
            with open(NF_TFIDF_PREPROCESSOR_PATH, "rb") as f: resources['tfidf_preprocessor'] = pickle.load(f)
        else: st.error(f"Missing: {NF_TFIDF_PREPROCESSOR_PATH}"); resources['tfidf_preprocessor'] = None

        if os.path.exists(NF_TABULAR_PREPROCESSOR_NN_PATH):
            with open(NF_TABULAR_PREPROCESSOR_NN_PATH, "rb") as f:
                loaded_tabular_prep = pickle.load(f)
                resources['tabular_preprocessor_nn'] = None if isinstance(loaded_tabular_prep, str) else loaded_tabular_prep # Handle potential placeholder string
        else: st.sidebar.warning(f"NN Tabular Preprocessor not found: {NF_TABULAR_PREPROCESSOR_NN_PATH}"); resources['tabular_preprocessor_nn'] = None

        if os.path.exists(NF_RF_MODEL_PATH):
            with open(NF_RF_MODEL_PATH, "rb") as f: resources['rf_model'] = pickle.load(f)
        else: st.error(f"Missing: {NF_RF_MODEL_PATH}"); resources['rf_model'] = None
        
        X_test_tab, X_test_bert, y_test_data = None, None, None
        if os.path.exists(NF_X_TEST_TABULAR_NN_PATH):
            with open(NF_X_TEST_TABULAR_NN_PATH, "rb") as f: X_test_tab = pickle.load(f)
        if os.path.exists(NF_X_TEST_BERT_EMBEDDINGS_PATH):
             with open(NF_X_TEST_BERT_EMBEDDINGS_PATH, "rb") as f: X_test_bert = pickle.load(f)
        if os.path.exists(NF_Y_TEST_PATH):
            with open(NF_Y_TEST_PATH, "rb") as f: y_test_data = pickle.load(f)

        resources['X_test_tabular_nn'] = X_test_tab
        resources['X_test_bert_embeddings'] = X_test_bert
        resources['y_test'] = y_test_data
        
        resources['nn_model'] = None
        # Try loading .keras model first (preferred)
        if os.path.exists(NF_NN_MODEL_KERAS_PATH):
            try:
                resources['nn_model'] = tf.keras.models.load_model(NF_NN_MODEL_KERAS_PATH, compile=False)
                nn_model_loaded_flag = True
                st.sidebar.success("NN Model: Loaded from .keras format.")
            except Exception as e_keras:
                st.sidebar.warning(f"Failed to load NN from .keras: {e_keras}. Trying .h5 weights.")
        
        # Fallback to building model and loading weights from .h5 if .keras failed or not present
        if not nn_model_loaded_flag and X_test_tab is not None and X_test_bert is not None and y_test_data is not None:
            if os.path.exists(NF_NN_MODEL_H5_PATH): # Assuming .h5 contains weights if .keras is not used
                try:
                    input_dim = X_test_tab.shape[1] + X_test_bert.shape[1]
                    num_classes = pd.Series(y_test_data).nunique() # Ensure y_test_data is Series for nunique
                    nn_model_rc = nf_build_nn_model(input_dim, num_classes)
                    nn_model_rc.load_weights(NF_NN_MODEL_H5_PATH) # Load weights
                    resources['nn_model'] = nn_model_rc
                    nn_model_loaded_flag = True
                    st.sidebar.success("NN Model: Built & Weights loaded from .h5.")
                except Exception as e_weights:
                    st.sidebar.error(f"Failed to load NN weights from {NF_NN_MODEL_H5_PATH}: {e_weights}")
            else:
                st.sidebar.warning(f"NN Weights file ({NF_NN_MODEL_H5_PATH}) not found. Cannot build NN via weights.")
        elif not nn_model_loaded_flag: # If test data for shape inference was missing
             st.sidebar.warning("NN Model: Cannot build via weights (missing test data for shape).")


        # Fallback to loading full .h5 model if others failed (legacy)
        if not nn_model_loaded_flag and os.path.exists(NF_NN_MODEL_H5_PATH):
            try:
                # Check if it's a full model or just weights by trying to load as full model
                temp_model = tf.keras.models.load_model(NF_NN_MODEL_H5_PATH, compile=False)
                resources['nn_model'] = temp_model
                nn_model_loaded_flag = True
                st.sidebar.info("NN Model: Loaded from .h5 format (full model).")
            except Exception as e_h5_full: # It was likely just weights or incompatible
                st.sidebar.warning(f"Interpreting {NF_NN_MODEL_H5_PATH} as full model failed: {e_h5_full}. Assuming it's weights-only if build succeeded.")


        if not nn_model_loaded_flag:
            st.sidebar.error("NN Model: Could not be loaded.")

        resources['nn_model_loaded_successfully'] = nn_model_loaded_flag

        resources['bert_tokenizer'] = DistilBertTokenizer.from_pretrained(NF_BERT_MODEL_NAME)
        resources['bert_model'] = TFDistilBertModel.from_pretrained(NF_BERT_MODEL_NAME)
        resources['bert_model'].trainable = False # Ensure it's not trainable for inference

        if os.path.exists(NF_X_TEST_TFIDF_PATH):
            with open(NF_X_TEST_TFIDF_PATH, "rb") as f: resources['X_test_tfidf'] = pickle.load(f)
        else: st.error(f"Missing: {NF_X_TEST_TFIDF_PATH}"); resources['X_test_tfidf'] = None

    except Exception as e:
        st.error(f"An unexpected error occurred while loading Recall Predictor resources: {e}")
        st.exception(e)
    return resources

def nf_generate_bert_embeddings(texts, tokenizer, model, max_len=128):
    if not texts or (isinstance(texts, list) and len(texts) == 1 and not texts[0]): # Handle empty or list of empty string
        if model and hasattr(model.config, 'hidden_size'):
             return np.zeros((len(texts), model.config.hidden_size))
        else: # Fallback if model or config is not as expected
             return np.zeros((len(texts), 768)) # Default DistilBERT hidden size

    # Ensure texts is a list of strings
    if isinstance(texts, pd.Series): texts = texts.tolist()
    if not isinstance(texts, list): texts = [str(texts)] # Convert single non-list item to list
    texts = [str(t) if pd.notna(t) else "" for t in texts] # Ensure all are strings, handle NaN

    inputs = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf')
    outputs = model(inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


# --- Main Application Logic ---

def render_brand_analyzer():
    st.title("🔬 CFIA Recall - Brand Association Analyzer")

    # --- User Inputs (Sidebar for Brand Analyzer) ---
    st.sidebar.header("Brand Analyzer Settings")
    data_path = st.sidebar.text_input("Path to CFIA analysis output directory:", GT_DEFAULT_DATA_PATH, key="gt_data_path")
    connections_file_path = os.path.join(data_path, GT_CONNECTIONS_FILE)
    ml_data_file_path = os.path.join(data_path, GT_ML_DATA_FILE)

    brand_graph, max_graph_weight = gt_load_brand_graph(connections_file_path)
    ml_df_gt = gt_load_ml_data(ml_data_file_path) # Renamed to avoid clash with nf_resources['full_df']

    if brand_graph is not None and ml_df_gt is not None:
        st.sidebar.success("Brand Analyzer Data loaded successfully!")
        tfidf_vectorizer_gt, product_tfidf_matrix_gt, all_product_names_for_tfidf_gt = gt_get_tfidf_vectorizer_and_matrix(ml_df_gt)
        brand_profiles_df_gt = gt_calculate_brand_profiles(ml_df_gt)
        brand_profiles_clustered_df_gt = gt_get_brand_clusters(brand_profiles_df_gt) if brand_profiles_df_gt is not None else None
        
        available_brands_gt = []
        if brand_profiles_clustered_df_gt is not None and brand_graph is not None:
            available_brands_gt = sorted([b for b in brand_profiles_clustered_df_gt.index if b in brand_graph.nodes()])
        elif ml_df_gt is not None: # Fallback if clustering failed
             available_brands_gt = sorted(ml_df_gt['BRAND_NAME'].unique().tolist())


        unique_contaminants_gt = sorted(ml_df_gt['AREA_OF_CONCERN'].unique().tolist()) if ml_df_gt is not None else []

        st.sidebar.subheader("Search Parameters") # Changed from header to subheader
        selected_brand_input_gt = st.sidebar.selectbox("Select Primary Brand:", options=available_brands_gt, index=0 if available_brands_gt else -1, key="gt_brand_select")
        input_product_query_val_gt = st.sidebar.text_input("Product Name (Optional):", key="gt_product_query").lower().strip()
        # Ensure options for contaminant selectbox are strings and add an empty option for "any"
        contaminant_options_gt = [""] + [str(c) for c in unique_contaminants_gt]
        input_contaminant_query_val_gt = st.sidebar.selectbox("Contaminant (Optional):", options=contaminant_options_gt, key="gt_contaminant_query").lower().strip()

        top_n_suggestions_val_gt = st.sidebar.slider("Number of Suggestions:", 1, 50, GT_TOP_N_DEFAULT, key="gt_top_n")
        product_similarity_threshold_val_gt = st.sidebar.slider("Product Similarity Threshold:", 0.0, 1.0, GT_PRODUCT_SIMILARITY_THRESHOLD, 0.05, key="gt_sim_thresh")

        # --- Tabs for Brand Analyzer ---
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyzer", "🔗 Indirect Connections", "📊 Brand Profiler", "🧪 Examples"])

        with tab1:
            st.header("Direct Association Analyzer")
            if st.button("Find Direct Associations", type="primary", key="gt_find_direct"):
                if not selected_brand_input_gt:
                    st.warning("Please select a primary brand.")
                else:
                    st.subheader(f"Direct Associations for Brand: '{selected_brand_input_gt.title()}'")
                    graph_dot = gt_generate_graph_viz(brand_graph, selected_brand_input_gt)
                    if graph_dot:
                        with st.expander("Show Brand Connections Graph"): st.graphviz_chart(graph_dot)
                    
                    results_df_gt = gt_run_association_analysis(
                        selected_brand_input_gt, input_product_query_val_gt, input_contaminant_query_val_gt, brand_graph, max_graph_weight, ml_df_gt,
                        tfidf_vectorizer_gt, product_tfidf_matrix_gt, all_product_names_for_tfidf_gt, product_similarity_threshold_val_gt, top_n_suggestions_val_gt)
                    
                    if not results_df_gt.empty:
                        cols_to_display = ["Score", "Associated Brand", "Recalled Product", "Original Contaminant", "Recall Class", "Recall Date", "Connection Weight", "Product Similarity"]
                        st.dataframe(results_df_gt[[c for c in cols_to_display if c in results_df_gt.columns]], hide_index=True, use_container_width=True)
                    else:
                        st.info("No direct matching associations found based on current filters.")
        with tab2:
            st.header("Indirect Connections (Friends of Friends)")
            st.markdown("Finds brands linked via one intermediary (A -> B -> C).")
            if st.button("Find Indirect Connections", key="gt_find_indirect"):
                if not selected_brand_input_gt:
                    st.warning("Please select a primary brand.")
                else:
                    st.subheader(f"Indirect Connections for Brand: '{selected_brand_input_gt.title()}'")
                    indirect_df_gt = gt_find_indirect_connections(brand_graph, selected_brand_input_gt, top_n=top_n_suggestions_val_gt) # Use slider for top_n
                    if not indirect_df_gt.empty:
                        indirect_df_gt['Indirect Brand'] = indirect_df_gt['Indirect Brand'].str.title()
                        st.dataframe(indirect_df_gt, hide_index=True, use_container_width=True)
                    else:
                        st.info("No significant indirect connections found.")
        with tab3:
            st.header("Brand Profiler & Clustering")
            st.markdown("Analyzes historical recall behavior to create profiles and groups similar brands.")
            if brand_profiles_clustered_df_gt is not None and not brand_profiles_clustered_df_gt.empty:
                # Ensure selected_brand_input_gt is valid for profiling
                profile_brands_options = available_brands_gt # Use the same list as the main selector
                
                selected_brand_for_profile_gt = st.selectbox(
                    "Select Brand to Profile:", 
                    options=profile_brands_options, 
                    index=profile_brands_options.index(selected_brand_input_gt) if selected_brand_input_gt in profile_brands_options else 0,
                    key="gt_profile_brand_select"
                )
                
                if selected_brand_for_profile_gt and selected_brand_for_profile_gt in brand_profiles_clustered_df_gt.index:
                    st.subheader(f"Profile for: {selected_brand_for_profile_gt.title()}")
                    profile_gt = brand_profiles_clustered_df_gt.loc[selected_brand_for_profile_gt]
                    st.metric("Assigned Cluster", f"Cluster {int(profile_gt['Cluster'])}")
                    st.dataframe(profile_gt.apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_frame().T, hide_index=True)

                    st.subheader(f"Other Brands in Cluster {int(profile_gt['Cluster'])}")
                    cluster_brands_gt = brand_profiles_clustered_df_gt[brand_profiles_clustered_df_gt['Cluster'] == profile_gt['Cluster']].index.tolist()
                    if selected_brand_for_profile_gt in cluster_brands_gt: # remove self if present
                        cluster_brands_gt.remove(selected_brand_for_profile_gt) 
                    
                    if cluster_brands_gt:
                        # Displaying limited columns for brevity or select key profile metrics
                        display_cols_cluster = ['Total Recalls', 'Unique Incidents', 'Class I Pct', 'Top Contaminant']
                        actual_display_cols = [col for col in display_cols_cluster if col in brand_profiles_clustered_df_gt.columns]
                        
                        st.dataframe(
                            brand_profiles_clustered_df_gt.loc[cluster_brands_gt, actual_display_cols].reset_index().rename(columns={'index': 'Brand Name'}).applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x), 
                            hide_index=True, 
                            use_container_width=True
                        )
                    else:
                        st.info("No other brands in this cluster.")
                elif selected_brand_for_profile_gt:
                     st.warning(f"Profile for '{selected_brand_for_profile_gt}' not found in clustered data.")
            else:
                st.warning("Brand profiles or clustered data could not be generated or is empty.")

        with tab4:
            st.header("Examples of Multi-Brand Recalls")
            example_incidents_gt = gt_get_example_incidents(ml_df_gt)
            if not example_incidents_gt:
                st.warning("Could not find suitable multi-brand recall examples.")
            else:
                for i, ex in enumerate(example_incidents_gt):
                    expander_title = f"Example {i+1}: Recall #{ex['recall_number']} ({ex['primary_brand_example'].title() if ex['primary_brand_example'] else 'N/A'})"
                    with st.expander(expander_title):
                        st.markdown(f"**Original Incident:** Brands `{ex['primary_brand_example'].title()}` and `{', '.join([b.title() for b in ex['other_brands_involved']])}` were recalled for `{ex['example_contaminant'].title()}` around `{ex['recall_date']}`.")
                        st.dataframe(ex['full_incident_details_df'].rename(columns=lambda c: c.replace('_', ' ').title()), hide_index=True)
                        
                        if st.button(f"Analyze this Example Incident ({ex['primary_brand_example'].title()})", key=f"gt_example_{i}"):
                            st.markdown(f"**Analyzer Results for Example:**")
                            # Use example data for the analyzer function
                            example_results_df_gt = gt_run_association_analysis(
                                ex['primary_brand_example'], ex['example_product'], ex['example_contaminant'],
                                brand_graph, max_graph_weight, ml_df_gt, tfidf_vectorizer_gt, product_tfidf_matrix_gt, all_product_names_for_tfidf_gt,
                                product_similarity_threshold_val_gt, top_n_suggestions_val_gt) # Use current slider values for consistency
                            
                            if not example_results_df_gt.empty:
                                found = [b.title() for b in ex['other_brands_involved'] if b.lower() in example_results_df_gt['Associated Brand'].str.lower().values] # Case-insensitive check
                                if found: st.success(f"✅ Found known co-recalled brand(s) in suggestions: **{', '.join(found)}**")
                                else: st.warning(f"ℹ️ Known co-recalled brand(s) not found in top suggestions with current settings.")
                                
                                cols_to_display_ex = ["Score", "Associated Brand", "Recalled Product", "Original Contaminant", "Recall Class", "Recall Date"]
                                st.dataframe(example_results_df_gt[[c for c in cols_to_display_ex if c in example_results_df_gt.columns]], hide_index=True, use_container_width=True)
                            else:
                                st.info("No associations found for this example incident with current settings.")
    else:
        st.warning("Brand Analyzer data could not be loaded. Please check paths and ensure `cfia_brand_connections.csv` and `cfia_enhanced_dataset_ml.csv` exist in the specified directory.")
        st.markdown(f"Expected connections: `{connections_file_path}`")
        st.markdown(f"Expected ML data: `{ml_data_file_path}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Brand Analyzer V3 features.")


def render_recall_predictor():
    st.title("🇨🇦 CFIA Food Recall Class Predictor")
    st.markdown("Predict food recall severity using ML trained on CFIA data.")

    nf_resources = nf_load_resources() # Load all resources for this app part

    if not nf_resources or not nf_resources.get('full_df') is not None: # Check if essential full_df loaded
         st.error("Recall Predictor application cannot start: Essential resources (like full dataset) failed to load. Please check file paths and console errors.")
         st.markdown(f"Expected full dataset: `{NF_FULL_DATASET_PATH}`")
         return # Exit if essential data is missing

    # --- Sidebar for Recall Predictor ---
    st.sidebar.header("Recall Predictor Settings")
    nf_app_mode = st.sidebar.radio("Choose Mode:", ["🔮 Predict New Recall", "📊 Explore Test Data"], key="nf_app_mode")
    
    model_choice_disabled = not nf_resources.get('nn_model_loaded_successfully', False)
    
    nf_model_choice = st.sidebar.selectbox("Select Prediction Model:",
                                        ["Random Forest", "Neural Network (BERT + Tabular)"],
                                        key="nf_model_selector",
                                        disabled=model_choice_disabled)
    
    if model_choice_disabled and nf_model_choice == "Neural Network (BERT + Tabular)":
        st.sidebar.warning("NN model not available, defaulting to Random Forest for Recall Predictor.")
        nf_model_choice = "Random Forest"
    
    if nf_resources.get('rf_model') is None and nf_model_choice == "Random Forest":
        st.error("Random Forest model is not loaded. Predictions cannot be made with RF.")
        return


    if nf_app_mode == "🔮 Predict New Recall":
        st.header("🔮 Predict New Recall Class")
        all_cols_nf = nf_resources.get('original_training_columns', [])
        if not all_cols_nf:
            st.error("Original training columns not loaded. Cannot proceed with prediction.")
            return

        day_cols = [col for col in all_cols_nf if col.startswith('Day_')]
        season_cols = [col for col in all_cols_nf if col.startswith('Season_')]
        depth_cols = [col for col in all_cols_nf if col.startswith('Depth_')]
        
        depth_options_map = {
            'consumer': 'Depth_consumer', 'retail/hri': 'Depth_retail/hri', #hri = hotel/restaurant/institution
            'wholesale': 'Depth_wholesale', 'unknown': 'Depth_nan'
        }
        depth_display_options = sorted(list(depth_options_map.keys()))
        
        area_opts_nf = nf_resources.get('area_options', [])
        primary_opts_nf = nf_resources.get('primary_options', ["Yes", "No", "Unknown"])

        # Default indices, handling cases where 'unknown' might be missing
        unknown_area_index = area_opts_nf.index('unknown') if 'unknown' in area_opts_nf else 0
        unknown_primary_index = primary_opts_nf.index('Unknown') if 'Unknown' in primary_opts_nf else (primary_opts_nf.index('unknown') if 'unknown' in primary_opts_nf else 0)
        unknown_depth_index = depth_display_options.index('unknown') if 'unknown' in depth_display_options else 0


        with st.form(key="nf_new_recall_form"):
            st.subheader("📝 Essential Information")
            c1, c2 = st.columns(2)
            common_name_nf = c1.text_input("Product Common Name", "e.g., Organic Blueberries", key="nf_common_name")
            recall_date_nf = c2.date_input("Recall Date", datetime.today(), key="nf_recall_date")

            area_of_concern_nf = st.selectbox("Area of Concern", area_opts_nf, index=unknown_area_index, key="nf_area_of_concern")

            c3, c4 = st.columns(2)
            primary_recall_nf = c3.selectbox("Primary Recall?", primary_opts_nf, index=unknown_primary_index, key="nf_primary_recall").lower()
            depth_input_key_nf = c4.selectbox("Recall Depth", options=depth_display_options, index=unknown_depth_index, key="nf_depth")

            with st.expander("➕ Optional / Advanced Details (Defaults from Training Data Averages if available)"):
                # Try to get sensible defaults if full_df is loaded
                defaults = {}
                if nf_resources.get('full_df') is not None:
                    df_for_defaults = nf_resources['full_df']
                    num_cols_for_default = [
                        'DAYS_SINCE_LAST_BRAND_RECALL', 'BRAND_RECALL_FREQ', 'Incident_BrandsInvolved', 
                        'Incident_NumItems', 'DAYS_SINCE_LAST_PROD_RECALL', 'PATHOGEN_FREQ', 
                        'Incident_UniqueProducts', 'Incident_DurationDays'
                    ]
                    for col in num_cols_for_default:
                        if col in df_for_defaults.columns:
                            defaults[col] = int(df_for_defaults[col].median()) # Using median for robustness
                        else:
                            defaults[col] = 0 # Fallback default
                else: # Fallback defaults if full_df not available
                    defaults = {k: (30 if "DAYS" in k else 1) for k in num_cols_for_default}
                    defaults['Incident_DurationDays'] = 0


                co1, co2 = st.columns(2)
                days_since_brand = co1.number_input("Days Since Last Brand Recall", 0, value=defaults.get('DAYS_SINCE_LAST_BRAND_RECALL',30), key="nf_days_brand")
                brand_recall_freq = co1.number_input("Brand Recall Frequency", 0, value=defaults.get('BRAND_RECALL_FREQ',1), key="nf_brand_freq")
                incident_brands_involved = co1.number_input("# Brands in Incident", 1, value=defaults.get('Incident_BrandsInvolved',1), key="nf_inc_brands")
                incident_num_items = co1.number_input("# Items in Incident", 1, value=defaults.get('Incident_NumItems',1), key="nf_inc_items")
                
                days_since_prod = co2.number_input("Days Since Last Product Recall", 0, value=defaults.get('DAYS_SINCE_LAST_PROD_RECALL',90), key="nf_days_prod")
                pathogen_freq = co2.number_input("Pathogen/Concern Frequency", 0, value=defaults.get('PATHOGEN_FREQ',1), key="nf_path_freq")
                incident_unique_products = co2.number_input("# Unique Products in Incident", 1, value=defaults.get('Incident_UniqueProducts',1), key="nf_inc_prods")
                incident_duration_days = co2.number_input("Incident Duration (Days)", 0, value=defaults.get('Incident_DurationDays',0), key="nf_inc_duration")

            submit_button_nf = st.form_submit_button("🚀 Predict Class")

        if submit_button_nf and common_name_nf:
            input_data_nf = {}
            input_data_nf['COMMON_NAME'] = common_name_nf
            input_data_nf['AREA_OF_CONCERN'] = area_of_concern_nf.lower() # Ensure lowercase
            input_data_nf['PRIMARY_RECALL'] = primary_recall_nf.lower() # Ensure lowercase

            input_data_nf['DAYS_SINCE_LAST_BRAND_RECALL'] = days_since_brand
            input_data_nf['DAYS_SINCE_LAST_PROD_RECALL'] = days_since_prod
            input_data_nf['BRAND_RECALL_FREQ'] = brand_recall_freq
            input_data_nf['PATHOGEN_FREQ'] = pathogen_freq
            input_data_nf['Incident_BrandsInvolved'] = incident_brands_involved
            input_data_nf['Incident_UniqueProducts'] = incident_unique_products
            input_data_nf['Incident_NumItems'] = incident_num_items
            input_data_nf['Incident_DurationDays'] = incident_duration_days
            input_data_nf['YEAR'] = recall_date_nf.year
            input_data_nf['MONTH'] = recall_date_nf.month
            input_data_nf['WEEK'] = int(recall_date_nf.isocalendar().week)
            input_data_nf['QUARTER'] = (recall_date_nf.month - 1) // 3 + 1

            day_of_week_str = recall_date_nf.strftime("%A") # e.g., Monday
            season_str = ("Winter" if recall_date_nf.month in [12, 1, 2] else "Spring" if recall_date_nf.month in [3, 4, 5]
                          else "Summer" if recall_date_nf.month in [6, 7, 8] else "Fall")

            for col in day_cols: input_data_nf[col] = 1 if col == f'Day_{day_of_week_str}' else 0
            if f'Day_{day_of_week_str}' not in day_cols and 'Day_nan' in day_cols: input_data_nf['Day_nan'] = 1
            
            for col in season_cols: input_data_nf[col] = 1 if col == f'Season_{season_str}' else 0
            if f'Season_{season_str}' not in season_cols and 'Season_nan' in season_cols: input_data_nf['Season_nan'] = 1

            target_depth_col = depth_options_map[depth_input_key_nf]
            for col in depth_cols: input_data_nf[col] = 1 if col == target_depth_col else 0
            if target_depth_col not in depth_cols and 'Depth_nan' in depth_cols: input_data_nf['Depth_nan'] = 1
            
            input_df_nf = pd.DataFrame([input_data_nf])
            missing_cols = set(all_cols_nf) - set(input_df_nf.columns)
            for c in missing_cols: input_df_nf[c] = 0 
            input_df_nf = input_df_nf[all_cols_nf]

            for col in all_cols_nf:
                 if input_df_nf[col].dtype == 'bool' or col.startswith(('Day_','Season_','Depth_')):
                      input_df_nf[col] = input_df_nf[col].astype(float)
            
            st.markdown("---"); st.subheader("🔍 Prediction Results")
            try:
                with st.spinner(f"Predicting with {nf_model_choice}..."):
                    if nf_model_choice == "Random Forest":
                        if nf_resources.get('tfidf_preprocessor') and nf_resources.get('rf_model'):
                            features_rf_nf = nf_resources['tfidf_preprocessor'].transform(input_df_nf)
                            pred_nf = nf_resources['rf_model'].predict(features_rf_nf)[0]
                            proba_nf = nf_resources['rf_model'].predict_proba(features_rf_nf)[0]
                            classes_nf = nf_resources['rf_model'].classes_
                        else: st.error("RF model or preprocessor not loaded."); return
                    else: # Neural Network
                        if nf_resources.get('nn_model') and nf_resources.get('tabular_preprocessor_nn') and nf_resources.get('bert_tokenizer') and nf_resources.get('bert_model'):
                            bert_emb_nf = nf_generate_bert_embeddings(input_df_nf['COMMON_NAME'], nf_resources['bert_tokenizer'], nf_resources['bert_model'])
                            tab_feat_nf = nf_resources['tabular_preprocessor_nn'].transform(input_df_nf)
                            # Ensure dtypes are float32 for TF
                            comb_feat_nf = np.hstack((tab_feat_nf.astype(np.float32), bert_emb_nf.astype(np.float32)))
                            pred_probs_nn_nf = nf_resources['nn_model'].predict(comb_feat_nf)[0]
                            pred_nf = np.argmax(pred_probs_nn_nf)
                            proba_nf = pred_probs_nn_nf
                            classes_nf = np.arange(len(proba_nf)) # Classes are 0, 1, 2 for NN
                        else: st.error("NN model or its components not loaded."); return

                st.success(f"Predicted Recall Class: **{NF_LABEL_MAP.get(pred_nf, 'Unknown')}**")
                prob_df_nf = pd.DataFrame({"Class": [NF_LABEL_MAP.get(c, f"Class {c}") for c in classes_nf], "Probability": proba_nf}
                                       ).sort_values("Probability", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df_nf)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)
        elif submit_button_nf and not common_name_nf:
            st.warning("Product Common Name is required for Recall Prediction.")

    elif nf_app_mode == "📊 Explore Test Data":
        st.header("📊 Explore Test Data & Model Performance")
        required_test_data = ['y_test', 'full_df', 'X_test_tfidf', 'X_test_tabular_nn', 'X_test_bert_embeddings']
        if not all(nf_resources.get(k) is not None for k in required_test_data):
            st.warning("Test data, labels, or full dataset not fully loaded for Recall Predictor. Exploration might be limited.")
            missing_items = [k for k in required_test_data if nf_resources.get(k) is None]
            st.markdown(f"Missing items: `{', '.join(missing_items)}`")
        else:
            y_test_nf = nf_resources['y_test']
            idx_nf = st.slider("Test Sample Index:", 0, len(y_test_nf) - 1, 0, key="nf_test_slider")
            true_label_nf = y_test_nf.iloc[idx_nf] if hasattr(y_test_nf, 'iloc') else y_test_nf[idx_nf]
            st.subheader(f"🎯 True Class: {NF_LABEL_MAP.get(true_label_nf, 'Unknown')}")

            try: # Display original features
                original_data_index = y_test_nf.index[idx_nf] if hasattr(y_test_nf, 'index') else idx_nf
                st.caption("Original Features (from cfia_enhanced_dataset_ml.csv):")
                st.dataframe(nf_resources['full_df'].iloc[[original_data_index]])
            except Exception as e: st.caption(f"Could not map to original full_df row reliably: {e}")
            st.markdown("---")

            if nf_model_choice == "Random Forest":
                if nf_resources.get('X_test_tfidf') is not None and nf_resources.get('rf_model') is not None:
                    sample_rf_nf = nf_resources['X_test_tfidf'][idx_nf].reshape(1,-1)
                    pred_rf_nf = nf_resources['rf_model'].predict(sample_rf_nf)[0]
                    st.markdown(f"🤖 **RF Prediction:** **{NF_LABEL_MAP.get(pred_rf_nf, 'Unknown')}** {'✅ Correct' if pred_rf_nf == true_label_nf else '❌ Incorrect'}")
                else: st.error("RF test data or model not loaded for exploration.")
            else: # Neural Network
                if nf_resources.get('X_test_tabular_nn') is not None and \
                   nf_resources.get('X_test_bert_embeddings') is not None and \
                   nf_resources.get('nn_model') is not None:
                    
                    tab_sample_nf = nf_resources['X_test_tabular_nn'][idx_nf].reshape(1,-1).astype(np.float32)
                    bert_sample_nf = nf_resources['X_test_bert_embeddings'][idx_nf].reshape(1,-1).astype(np.float32)
                    comb_sample_nf = np.hstack((tab_sample_nf, bert_sample_nf))
                    pred_nn_nf = np.argmax(nf_resources['nn_model'].predict(comb_sample_nf)[0])
                    st.markdown(f"🧠 **NN Prediction:** **{NF_LABEL_MAP.get(pred_nn_nf, 'Unknown')}** {'✅ Correct' if pred_nn_nf == true_label_nf else '❌ Incorrect'}")
                else: st.error("NN test data or model not loaded for exploration.")

            if st.button("Show Confusion Matrix (Full Test Set)", key="nf_show_cm_btn"):
                fig, ax = plt.subplots()
                y_true_cm_nf = nf_resources['y_test']
                try:
                    if nf_model_choice == "Random Forest":
                        if nf_resources.get('X_test_tfidf') is not None and nf_resources.get('rf_model') is not None:
                            y_pred_cm_nf = nf_resources['rf_model'].predict(nf_resources['X_test_tfidf'])
                            ax.set_title("Random Forest Confusion Matrix")
                        else: raise ValueError("RF test data or model missing for CM.")
                    else: # Neural Network
                        if nf_resources.get('X_test_tabular_nn') is not None and \
                           nf_resources.get('X_test_bert_embeddings') is not None and \
                           nf_resources.get('nn_model') is not None:
                            
                            tab_all_nf = nf_resources['X_test_tabular_nn'].astype(np.float32)
                            bert_all_nf = nf_resources['X_test_bert_embeddings'].astype(np.float32)
                            comb_all_nf = np.hstack((tab_all_nf, bert_all_nf))
                            y_pred_cm_nf = np.argmax(nf_resources['nn_model'].predict(comb_all_nf), axis=1)
                            ax.set_title("Neural Network Confusion Matrix")
                        else: raise ValueError("NN test data or model missing for CM.")

                    cm_nf = confusion_matrix(y_true_cm_nf, y_pred_cm_nf, labels=list(NF_LABEL_MAP.keys()))
                    sns.heatmap(cm_nf, annot=True, fmt="d", cmap="Blues", xticklabels=NF_LABEL_MAP.values(), yticklabels=NF_LABEL_MAP.values(), ax=ax)
                    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                    st.pyplot(fig)
                except Exception as e: st.error(f"Confusion Matrix Error: {e}")
                finally: 
                    plt.close(fig) # Close the figure to free memory
                    gc.collect() # Explicitly collect garbage

    st.sidebar.markdown("---")
    st.sidebar.info("Recall Predictor: Educational tool based on CFIA data.")


# --- Main App Router ---
st.sidebar.title("App Navigation")
app_selection = st.sidebar.radio(
    "Choose an Application:",
    ("Brand Association Analyzer", "Recall Class Predictor"),
    key="main_app_selector"
)

if app_selection == "Brand Association Analyzer":
    render_brand_analyzer()
elif app_selection == "Recall Class Predictor":
    render_recall_predictor()
