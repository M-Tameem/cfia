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
import graphviz # For visualization
import math

# --- Configuration ---
DEFAULT_DATA_PATH = './cfia_analysis_output_v3/' # Default path to your script's output
CONNECTIONS_FILE = 'cfia_brand_connections.csv'
ML_DATA_FILE = 'cfia_enhanced_dataset_ml.csv'
PRODUCT_SIMILARITY_THRESHOLD = 0.3 # Adjust as needed
TOP_N_DEFAULT = 10
NUM_EXAMPLES_TO_SHOW = 5 # For the Examples tab
NUM_CLUSTERS = 6 # K for K-Means clustering (adjust as needed)

# Ranking Weights (tune these as needed)
W_CONN = 0.50           # Weight for brand connection strength
W_PROD_SIM = 0.30       # Weight for product similarity
W_CONT_MATCH = 0.15     # Bonus for matching contaminant
W_PRIMARY_RECALL = 0.05 # Bonus if it's a primary recall

# --- Helper Functions ---

@st.cache_resource # Cache the graph resource
def load_brand_graph(connections_path):
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

@st.cache_data # Cache the DataFrame
def load_ml_data(ml_data_path):
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
        # Ensure incident cols are numeric, fillna with 0 or 1 as appropriate
        for col in ['Incident_NumItems', 'Incident_BrandsInvolved', 'Incident_UniqueProducts', 'Incident_DurationDays']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading ML dataset: {e}")
        return None

@st.cache_resource
def get_tfidf_vectorizer_and_matrix(_ml_df):
    if _ml_df is None: return None, None, []
    product_names = _ml_df['COMMON_NAME'].unique().tolist()
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(product_names)
    return vectorizer, tfidf_matrix, product_names

def get_product_similarity(query_product, vectorizer, product_tfidf_matrix, all_product_names):
    if not query_product or vectorizer is None: return {}
    query_vector = vectorizer.transform([query_product.lower().strip()])
    cosine_similarities = cosine_similarity(query_vector, product_tfidf_matrix).flatten()
    return {name: score for name, score in zip(all_product_names, cosine_similarities)}

@st.cache_data
def calculate_brand_profiles(_ml_df):
    """Calculates risk profiles for each brand."""
    if _ml_df is None: return None
    
    profiles = {}
    brand_groups = _ml_df.groupby('BRAND_NAME')

    for brand, group in brand_groups:
        if not brand or brand == 'unknown': continue
        
        class_counts = group['Incident_Severity'].value_counts(normalize=True) * 100
        contaminant_counts = group['AREA_OF_CONCERN'].value_counts()
        
        profiles[brand] = {
            'Total Recalls': len(group.drop_duplicates(subset=['RECALL_NUMBER', 'COMMON_NAME'])), # Count unique product-incident pairs
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
def get_brand_clusters(_profiles_df, n_clusters=NUM_CLUSTERS):
    """Applies K-Means clustering to brand profiles."""
    if _profiles_df is None or _profiles_df.empty: return None

    # Select numerical features for clustering
    features = [
        'Total Recalls', 'Unique Incidents', 'Class I Pct', 'Class II Pct', 
        'Unique Products', 'Avg Incident Size', 'Avg Brands Involved', 'Multi-Brand Rate'
    ]
    
    # Ensure all features exist
    features_to_use = [f for f in features if f in _profiles_df.columns]
    if not features_to_use: return None

    X = _profiles_df[features_to_use].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to profiles
    _profiles_df_clustered = _profiles_df.copy()
    _profiles_df_clustered['Cluster'] = cluster_labels
    return _profiles_df_clustered

@st.cache_data
def find_indirect_connections(_brand_graph, selected_brand, top_n=10):
    """Finds second-order connections and ranks them."""
    if not selected_brand or selected_brand not in _brand_graph:
        return pd.DataFrame()

    indirect = defaultdict(float)
    direct_neighbors = set(_brand_graph.neighbors(selected_brand)) if selected_brand in _brand_graph else set()

    for neighbor in direct_neighbors:
        weight1 = _brand_graph[selected_brand][neighbor].get('weight', 0)
        for second_neighbor in _brand_graph.neighbors(neighbor):
            # Exclude self and direct neighbors
            if second_neighbor != selected_brand and second_neighbor not in direct_neighbors:
                weight2 = _brand_graph[neighbor][second_neighbor].get('weight', 0)
                # Score: sum of (w1 * w2) for all paths
                indirect[second_neighbor] += weight1 * weight2
    
    if not indirect:
        return pd.DataFrame()
        
    indirect_df = pd.DataFrame(list(indirect.items()), columns=['Indirect Brand', 'Connection Score'])
    return indirect_df.sort_values('Connection Score', ascending=False).head(top_n).reset_index(drop=True)


# --- Functions from previous version (slightly adapted) ---
def calculate_suggestion_score(suggestion_details, input_contaminant_query, max_connection_weight):
    score = 0
    norm_conn_weight = (suggestion_details.get("Connection Weight", 0) / max_connection_weight) if max_connection_weight else 0
    score += norm_conn_weight * W_CONN
    prod_sim = suggestion_details.get("Product Similarity", 0.0)
    if not isinstance(prod_sim, (float, int)): prod_sim = 0.0
    score += prod_sim * W_PROD_SIM
    if input_contaminant_query and suggestion_details.get("Original Contaminant", "").lower() == input_contaminant_query.lower():
        score += 1.0 * W_CONT_MATCH
    if 'PRIMARY_RECALL' in suggestion_details and str(suggestion_details.get("PRIMARY_RECALL", "")).lower() == 'yes':
        score += 1.0 * W_PRIMARY_RECALL
    return round(score, 4)

def generate_graph_viz(brand_graph, selected_brand):
    if not selected_brand or selected_brand not in brand_graph: return None
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    dot.node(selected_brand.title(), shape='ellipse', style='filled', color='skyblue')
    max_neighbors_to_show = 15
    neighbors_count = 0
    for neighbor in brand_graph.neighbors(selected_brand):
        if neighbors_count >= max_neighbors_to_show:
            dot.node("...", label="...", shape='plaintext'); break
        weight = brand_graph[selected_brand][neighbor].get('weight', 0)
        dot.node(neighbor.title(), shape='ellipse')
        dot.edge(selected_brand.title(), neighbor.title(), label=f'w: {weight}')
        neighbors_count +=1
    return dot if neighbors_count > 0 else None

def run_association_analysis(selected_brand, input_product_query, input_contaminant_query,
                             brand_graph, max_graph_weight, ml_df,
                             tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf,
                             product_similarity_threshold_input, top_n_suggestions):
    # (This function remains largely the same as V2, but uses the new scoring)
    if not selected_brand: return pd.DataFrame()
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
        if input_contaminant_query: brand_recalls_df = brand_recalls_df[brand_recalls_df['AREA_OF_CONCERN'] == input_contaminant_query]
        if brand_recalls_df.empty: continue
        product_similarities_for_brand = get_product_similarity(input_product_query, tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf) if input_product_query and tfidf_vectorizer else {}
        for _, recall_row in brand_recalls_df.iterrows():
            recalled_product_name = recall_row['COMMON_NAME']; similarity_score = 0.0
            passes_product_filter = not input_product_query
            if input_product_query and tfidf_vectorizer:
                similarity_score = product_similarities_for_brand.get(recalled_product_name, 0.0)
                if similarity_score >= product_similarity_threshold_input: passes_product_filter = True
            if passes_product_filter:
                recall_key = (assoc_brand, recalled_product_name, recall_row['AREA_OF_CONCERN'], recall_row.get('RECALL_NUMBER', _))
                if recall_key not in processed_recalls:
                    suggestion_details = {
                        "Associated Brand": assoc_brand.title(), "Recalled Product": recalled_product_name.title(),
                        "Original Contaminant": recall_row['AREA_OF_CONCERN'].title(), "Recall Class": recall_row.get('Incident_Severity', 'N/A'),
                        "Recall Date": pd.to_datetime(recall_row.get('RECALL_DATE')).strftime('%Y-%m-%d') if pd.notna(recall_row.get('RECALL_DATE')) else 'N/A',
                        "Recall Number": recall_row.get('RECALL_NUMBER', 'N/A'), "Connection Weight": connection_weight,
                        "Product Similarity": round(similarity_score, 3) if input_product_query else "N/A", "PRIMARY_RECALL": recall_row.get('PRIMARY_RECALL', 'N/A')}
                    suggestion_details["Score"] = calculate_suggestion_score(suggestion_details, input_contaminant_query, max_graph_weight)
                    suggestions_list.append(suggestion_details)
                    processed_recalls.add(recall_key)
    sorted_suggestions = sorted(suggestions_list, key=lambda x: x['Score'], reverse=True)
    return pd.DataFrame(sorted_suggestions[:top_n_suggestions]) if sorted_suggestions else pd.DataFrame()

@st.cache_data
def get_example_incidents(_ml_df, num_examples=NUM_EXAMPLES_TO_SHOW):
    # (This function remains the same as V2)
    if _ml_df is None: return []
    incident_groups = _ml_df.groupby('RECALL_NUMBER')
    multi_brand_incidents = []
    for recall_num, group in incident_groups:
        unique_brands = group['BRAND_NAME'].unique()
        if len(unique_brands) > 1:
            primary_brand = unique_brands[0]; other_brands = unique_brands[1:]
            example_row = group.iloc[0]
            multi_brand_incidents.append({
                "recall_number": recall_num, "primary_brand_example": primary_brand, "other_brands_involved": list(other_brands),
                "example_product": example_row['COMMON_NAME'], "example_contaminant": example_row['AREA_OF_CONCERN'],
                "incident_severity": example_row.get('Incident_Severity', 'N/A'),
                "recall_date": pd.to_datetime(example_row.get('RECALL_DATE')).strftime('%Y-%m-%d') if pd.notna(example_row.get('RECALL_DATE')) else 'N/A',
                "full_incident_details_df": group[['BRAND_NAME', 'COMMON_NAME', 'AREA_OF_CONCERN', 'Incident_Severity', 'RECALL_DATE']].drop_duplicates().reset_index(drop=True)})
            if len(multi_brand_incidents) >= num_examples: break
    return multi_brand_incidents

# --- Main Application UI ---
st.set_page_config(layout="wide", page_title="CFIA Recall Brand Association Analyzer (V3)")
st.title("🔬 CFIA Recall - Brand Association Analyzer (V3)")

# --- User Inputs (Sidebar) ---
data_path = st.sidebar.text_input("Path to CFIA analysis output directory:", DEFAULT_DATA_PATH)
connections_file_path = os.path.join(data_path, CONNECTIONS_FILE)
ml_data_file_path = os.path.join(data_path, ML_DATA_FILE)

brand_graph, max_graph_weight = load_brand_graph(connections_file_path)
ml_df = load_ml_data(ml_data_file_path)

if brand_graph is not None and ml_df is not None:
    st.sidebar.success("Data loaded successfully!")
    tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf = get_tfidf_vectorizer_and_matrix(ml_df)
    brand_profiles_df = calculate_brand_profiles(ml_df)
    brand_profiles_clustered_df = get_brand_clusters(brand_profiles_df)
    
    available_brands = sorted([b for b in brand_profiles_clustered_df.index if b in brand_graph.nodes()]) if brand_profiles_clustered_df is not None and brand_graph is not None else sorted(ml_df['BRAND_NAME'].unique().tolist())
    unique_contaminants = sorted(ml_df['AREA_OF_CONCERN'].unique().tolist())

    st.sidebar.header("Search Parameters")
    selected_brand_input = st.sidebar.selectbox("Select Primary Brand:", options=available_brands, index=0 if available_brands else -1)
    input_product_query_val = st.sidebar.text_input("Product Name (Optional):").lower().strip()
    input_contaminant_query_val = st.sidebar.selectbox("Contaminant (Optional):", options=[""] + unique_contaminants).lower().strip()
    top_n_suggestions_val = st.sidebar.slider("Number of Suggestions:", 1, 50, TOP_N_DEFAULT)
    product_similarity_threshold_val = st.sidebar.slider("Product Similarity Threshold:", 0.0, 1.0, PRODUCT_SIMILARITY_THRESHOLD, 0.05)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyzer", "🔗 Indirect Connections", "📊 Brand Profiler", "🧪 Examples"])

    with tab1: # Analyzer
        st.header("Direct Association Analyzer")
        if st.button("Find Direct Associations", type="primary", key="find_analyzer"):
            st.subheader(f"Direct Associations for Brand: '{selected_brand_input.title()}'")
            graph_dot = generate_graph_viz(brand_graph, selected_brand_input)
            if graph_dot:
                with st.expander("Show Brand Connections Graph"): st.graphviz_chart(graph_dot)
            results_df = run_association_analysis(
                selected_brand_input, input_product_query_val, input_contaminant_query_val, brand_graph, max_graph_weight, ml_df,
                tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf, product_similarity_threshold_val, top_n_suggestions_val)
            if not results_df.empty:
                cols_to_display = ["Score", "Associated Brand", "Recalled Product", "Original Contaminant", "Recall Class", "Recall Date", "Connection Weight", "Product Similarity"]
                st.dataframe(results_df[[c for c in cols_to_display if c in results_df.columns]], hide_index=True, use_container_width=True)
            else:
                st.info("No direct matching associations found.")

    with tab2: # Indirect Connections
        st.header("Indirect Connections (Friends of Friends)")
        st.markdown("Finds brands linked via one intermediary (A -> B -> C). This can reveal potential, less obvious relationships.")
        if st.button("Find Indirect Connections", key="find_indirect"):
            st.subheader(f"Indirect Connections for Brand: '{selected_brand_input.title()}'")
            indirect_df = find_indirect_connections(brand_graph, selected_brand_input)
            if not indirect_df.empty:
                indirect_df['Indirect Brand'] = indirect_df['Indirect Brand'].str.title()
                st.dataframe(indirect_df, hide_index=True, use_container_width=True)
            else:
                st.info("No significant indirect connections found.")

    with tab3: # Brand Profiler
        st.header("Brand Profiler & Clustering")
        st.markdown("Analyzes historical recall behavior to create profiles and groups similar brands.")
        if brand_profiles_clustered_df is not None:
            selected_brand_for_profile = st.selectbox("Select Brand to Profile:", options=available_brands, index=available_brands.index(selected_brand_input) if selected_brand_input in available_brands else 0)
            
            if selected_brand_for_profile:
                st.subheader(f"Profile for: {selected_brand_for_profile.title()}")
                profile = brand_profiles_clustered_df.loc[selected_brand_for_profile]
                st.metric("Assigned Cluster", f"Cluster {int(profile['Cluster'])}")
                
                # Display profile details
                st.dataframe(profile.apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_frame().T, hide_index=True)

                st.subheader(f"Other Brands in Cluster {int(profile['Cluster'])}")
                cluster_brands = brand_profiles_clustered_df[brand_profiles_clustered_df['Cluster'] == profile['Cluster']].index.tolist()
                cluster_brands.remove(selected_brand_for_profile) # Remove self
                if cluster_brands:
                    st.dataframe(brand_profiles_clustered_df.loc[cluster_brands].reset_index().rename(columns={'index': 'Brand Name'}).applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x), hide_index=True, use_container_width=True)
                else:
                    st.info("No other brands in this cluster.")
        else:
            st.warning("Brand profiles could not be generated.")

    with tab4: # Examples
        st.header("Examples of Multi-Brand Recalls")
        example_incidents = get_example_incidents(ml_df)
        if not example_incidents:
            st.warning("Could not find suitable multi-brand recall examples.")
        else:
            for i, ex in enumerate(example_incidents):
                with st.expander(f"Example {i+1}: Recall #{ex['recall_number']} ({ex['primary_brand_example'].title()})"):
                    st.markdown(f"**Original Incident:** Brands `{ex['primary_brand_example'].title()}` and `{', '.join([b.title() for b in ex['other_brands_involved']])}` were recalled for `{ex['example_contaminant'].title()}` around `{ex['recall_date']}`.")
                    st.dataframe(ex['full_incident_details_df'].rename(columns=lambda c: c.replace('_', ' ').title()), hide_index=True)
                    if st.button(f"Analyze Example {i+1}", key=f"example_{i}"):
                        st.markdown(f"**Analyzer Results for Example:**")
                        example_results_df = run_association_analysis(
                            ex['primary_brand_example'], ex['example_product'], ex['example_contaminant'],
                            brand_graph, max_graph_weight, ml_df, tfidf_vectorizer, product_tfidf_matrix, all_product_names_for_tfidf,
                            product_similarity_threshold_val, top_n_suggestions_val)
                        if not example_results_df.empty:
                            found = [b.title() for b in ex['other_brands_involved'] if b.title() in example_results_df['Associated Brand'].values]
                            if found: st.success(f"✅ Found known brand(s): **{', '.join(found)}**")
                            else: st.warning(f"ℹ️ Known brand(s) not found.")
                            cols_to_display = ["Score", "Associated Brand", "Recalled Product", "Original Contaminant", "Recall Class", "Recall Date"]
                            st.dataframe(example_results_df[[c for c in cols_to_display if c in example_results_df.columns]], hide_index=True, use_container_width=True)
                        else:
                            st.info("No associations found for this example.")
else:
    st.warning("Data could not be loaded. Please check paths and run `cfia_advanced_analysis_v3.py`.")

st.sidebar.markdown("---")
st.sidebar.markdown("V3 - Added Profiling & Indirect Links.")
