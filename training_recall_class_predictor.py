import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc # Garbage Collection
import os # For checking file existence
import pickle # For saving/loading objects
from collections import Counter

# --- Install Transformers (if not already installed in the environment) ---
# !pip install transformers -q

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam # Explicit import

# Transformers imports
from transformers import DistilBertTokenizer, TFDistilBertModel

# --- Configuration & Setup ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_FILE = 'cfia_enhanced_dataset_ml.csv'
BERT_MODEL_NAME = 'distilbert-base-uncased'
BERT_BATCH_SIZE = 16
MAX_LEN = 128

# --- Filepaths for saved objects (specific to this recall class prediction task) ---
OUTPUT_DIR_RECALL_CLASS = "preprocessed_data_recall_class"
os.makedirs(OUTPUT_DIR_RECALL_CLASS, exist_ok=True)

TFIDF_PREPROCESSOR_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "tfidf_preprocessor_rc.pkl")
X_TRAIN_TFIDF_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_train_tfidf_rc.pkl")
X_TEST_TFIDF_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_test_tfidf_rc.pkl")
BERT_TRAIN_EMBEDDINGS_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_train_bert_embeddings_rc.pkl")
BERT_TEST_EMBEDDINGS_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_test_bert_embeddings_rc.pkl")
TABULAR_PREPROCESSOR_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "tabular_preprocessor_rc.pkl")
X_TRAIN_TABULAR_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_train_tabular_rc.pkl")
X_TEST_TABULAR_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "X_test_tabular_rc.pkl")

# --- 1. Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Successfully loaded {DATA_FILE} with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please upload it or check the path.")
    exit()

# --- 2. Target Definition & Initial Cleaning (RECALL_CLASS_NUM) ---
TARGET_COLUMN = 'RECALL_CLASS_NUM'
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found!")
    if 'RECALL_CLASS' in df.columns:
        print("Attempting to create RECALL_CLASS_NUM from RECALL_CLASS...")
        class_map = {'Class I': 1, 'Class II': 2, 'Class III': 3, 'Unknown': 0}
        df['RECALL_CLASS'] = df['RECALL_CLASS'].fillna('Unknown')
        df[TARGET_COLUMN] = df['RECALL_CLASS'].map(class_map).fillna(0)
    else:
        print("Cannot define target. Exiting.")
        exit()
df.dropna(subset=[TARGET_COLUMN], inplace=True)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
if 0 not in df[TARGET_COLUMN].unique() and df[TARGET_COLUMN].min() == 1:
    print("Adjusting target: Mapping [1, 2, 3,...] to [0, 1, 2,...] for NN compatibility.")
    unique_sorted_targets = sorted(df[TARGET_COLUMN].unique())
    target_map = {val: i for i, val in enumerate(unique_sorted_targets)}
    y = df[TARGET_COLUMN].map(target_map)
    print(f"Target mapping: {target_map}")
else:
    y = df[TARGET_COLUMN]
num_classes = y.nunique()
print(f"\nTarget variable '{TARGET_COLUMN}' (processed) created with {num_classes} classes.")
print(f"Target value counts:\n{y.value_counts(normalize=True) * 100}")
TEXT_FEATURE_COLUMN = 'COMMON_NAME'
df[TEXT_FEATURE_COLUMN] = df[TEXT_FEATURE_COLUMN].fillna('').astype(str)
if 'AREA_OF_CONCERN' in df.columns:
    df['AREA_OF_CONCERN'] = df['AREA_OF_CONCERN'].fillna('Unknown').astype(str).str.lower().str.strip()

# --- Define Features ---
print("\n--- Defining Feature Lists ---")
columns_to_drop = [
    'RECALL_DATE', 'RECALL_NUMBER', 'BRAND_NAME', 'RECALL_CLASS',
    'Incident_Pathogen', 'Incident_Severity', 'YEAR_MONTH', 'YEAR_SEASON',
] + [col for col in df.columns if col.startswith('Class_')]
columns_to_drop = list(set([col for col in columns_to_drop if col in df.columns]))
print(f"Initial columns_to_drop: {columns_to_drop}")

numerical_features_base = [
    'DAYS_SINCE_LAST_BRAND_RECALL', 'DAYS_SINCE_LAST_PROD_RECALL',
    'BRAND_RECALL_FREQ', 'PATHOGEN_FREQ', 'Incident_BrandsInvolved',
    'Incident_UniqueProducts', 'Incident_NumItems', 'Incident_DurationDays',
    'YEAR', 'MONTH', 'WEEK', 'QUARTER'
]
categorical_features_for_ohe_base = ['PRIMARY_RECALL', 'AREA_OF_CONCERN']
pre_one_hot_encoded_features_base = [col for col in df.columns if
                                     (col.startswith('Season_') or col.startswith('Day_') or col.startswith('Depth_'))]

# Filter for existence and ensure they are not target, text, or in columns_to_drop
numerical_features = [col for col in numerical_features_base if col in df.columns and col not in columns_to_drop and col != TARGET_COLUMN and col != TEXT_FEATURE_COLUMN]
categorical_features_for_ohe = [col for col in categorical_features_for_ohe_base if col in df.columns and col not in columns_to_drop and col != TARGET_COLUMN and col != TEXT_FEATURE_COLUMN]
pre_one_hot_encoded_features = [col for col in pre_one_hot_encoded_features_base if col in df.columns and col not in columns_to_drop and col != TARGET_COLUMN and col != TEXT_FEATURE_COLUMN]

# Ensure AREA_OF_CONCERN is not dropped if it's a feature
if 'AREA_OF_CONCERN' in categorical_features_for_ohe and 'AREA_OF_CONCERN' in columns_to_drop:
    print(f"Removing 'AREA_OF_CONCERN' from columns_to_drop as it's a feature.")
    columns_to_drop.remove('AREA_OF_CONCERN')

print(f"Filtered numerical_features: {numerical_features}")
print(f"Filtered categorical_features_for_ohe: {categorical_features_for_ohe}")
print(f"Filtered pre_one_hot_encoded_features: {pre_one_hot_encoded_features}")

# Ensure no overlap between feature groups (this part should be fine but let's be explicit)
final_numerical_features = [f for f in numerical_features if f not in categorical_features_for_ohe and f not in pre_one_hot_encoded_features]
final_categorical_features_for_ohe = [f for f in categorical_features_for_ohe if f not in final_numerical_features and f not in pre_one_hot_encoded_features]
final_pre_one_hot_encoded_features = [f for f in pre_one_hot_encoded_features if f not in final_numerical_features and f not in final_categorical_features_for_ohe]

print(f"Final numerical_features: {final_numerical_features}")
print(f"Final categorical_features_for_ohe: {final_categorical_features_for_ohe}") # CRITICAL: Should contain AREA_OF_CONCERN
print(f"Final pre_one_hot_encoded_features: {final_pre_one_hot_encoded_features}")

X_df = df.drop(columns=columns_to_drop + [TARGET_COLUMN])
if TARGET_COLUMN in X_df.columns: X_df = X_df.drop(columns=[TARGET_COLUMN])
for col in final_categorical_features_for_ohe: # Use final list for filling NaNs
    if col in X_df.columns: X_df[col] = X_df[col].fillna('Unknown')

# --- 3. Train-Test Split ---
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=SEED, stratify=y
)

# --- 4. TF-IDF Path (for RF Benchmark) ---
print("\n--- Processing for Random Forest (TF-IDF) - Recall Class ---")
X_train_tfidf, X_test_tfidf, tfidf_preprocessor = None, None, None
if os.path.exists(TFIDF_PREPROCESSOR_RC_PATH) and \
   os.path.exists(X_TRAIN_TFIDF_RC_PATH) and \
   os.path.exists(X_TEST_TFIDF_RC_PATH):
    print("Loading preprocessed TF-IDF data for Recall Class...")
    try:
        with open(TFIDF_PREPROCESSOR_RC_PATH, 'rb') as f: tfidf_preprocessor = pickle.load(f)
        with open(X_TRAIN_TFIDF_RC_PATH, 'rb') as f: X_train_tfidf = pickle.load(f)
        with open(X_TEST_TFIDF_RC_PATH, 'rb') as f: X_test_tfidf = pickle.load(f)
        print("TF-IDF data for Recall Class loaded successfully.")
    except Exception as e:
        print(f"Error loading TF-IDF Recall Class data: {e}. Regenerating...")
        X_train_tfidf = None
if 'X_train_tfidf' not in locals() or X_train_tfidf is None:
    print("Preprocessing TF-IDF data for Recall Class...")
    tfidf_transformers_list = [
        ('tfidf', TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2), stop_words='english'), TEXT_FEATURE_COLUMN)
    ]
    # Use final lists for ColumnTransformer
    if final_numerical_features: tfidf_transformers_list.append(('num', StandardScaler(), final_numerical_features))
    if final_categorical_features_for_ohe: tfidf_transformers_list.append(('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True), final_categorical_features_for_ohe))
    if final_pre_one_hot_encoded_features: tfidf_transformers_list.append(('passthrough_pre_ohe', 'passthrough', final_pre_one_hot_encoded_features))

    tfidf_preprocessor = ColumnTransformer(transformers=tfidf_transformers_list, remainder='drop')
    X_train_tfidf = tfidf_preprocessor.fit_transform(X_train_df)
    X_test_tfidf = tfidf_preprocessor.transform(X_test_df)
    print("Saving preprocessed TF-IDF data for Recall Class...")
    try:
        with open(TFIDF_PREPROCESSOR_RC_PATH, 'wb') as f: pickle.dump(tfidf_preprocessor, f)
        print(f"Saved: {TFIDF_PREPROCESSOR_RC_PATH}")
        with open(X_TRAIN_TFIDF_RC_PATH, 'wb') as f: pickle.dump(X_train_tfidf, f)
        print(f"Saved: {X_TRAIN_TFIDF_RC_PATH}")
        with open(X_TEST_TFIDF_RC_PATH, 'wb') as f: pickle.dump(X_test_tfidf, f)
        print(f"Saved: {X_TEST_TFIDF_RC_PATH}")
    except Exception as e:
        print(f"Error saving TF-IDF Recall Class data: {e}")
print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")

# --- 5. Random Forest Model (Benchmark for Recall Class) ---
print("\n--- Training Random Forest Classifier (TF-IDF) - Recall Class ---")
rf_classifier_rc = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1)
rf_classifier_rc.fit(X_train_tfidf, y_train)
y_pred_rf_rc = rf_classifier_rc.predict(X_test_tfidf)
print("\nRandom Forest (Recall Class) - Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_rc):.4f}")
print(classification_report(y_test, y_pred_rf_rc, zero_division=0))

# --- 6. BERT Path (Feature Extraction for NN Recall Class) ---
print("\n--- Processing for Neural Network (BERT Extract) - Recall Class ---")
bert_tokenizer_global, bert_model_global = None, None
X_train_bert, X_test_bert = None, None
if not (os.path.exists(BERT_TRAIN_EMBEDDINGS_RC_PATH) and os.path.exists(BERT_TEST_EMBEDDINGS_RC_PATH)):
    print("Loading BERT model for embedding generation...")
    bert_tokenizer_global = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model_global = TFDistilBertModel.from_pretrained(BERT_MODEL_NAME)
    bert_model_global.trainable = False
def batch_tokenize(texts, local_tokenizer, max_len):
    return local_tokenizer(
        texts, max_length=max_len, padding='max_length', truncation=True, return_tensors='tf'
    )
def generate_bert_embeddings(texts, local_bert_model, local_tokenizer, max_len, batch_size):
    embeddings = []
    num_texts = len(texts)
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = batch_tokenize(batch_texts, local_tokenizer, max_len)
        outputs = local_bert_model(inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embeddings)
        if (i // batch_size + 1) % 10 == 0 or (i // batch_size + 1) == ((num_texts // batch_size) +1) :
            print(f"Processed BERT batch {i // batch_size + 1}/{(num_texts // batch_size) + 1}")
        gc.collect()
    return np.vstack(embeddings)
if os.path.exists(BERT_TRAIN_EMBEDDINGS_RC_PATH) and os.path.exists(BERT_TEST_EMBEDDINGS_RC_PATH):
    print("Loading pre-generated BERT embeddings for Recall Class...")
    try:
        with open(BERT_TRAIN_EMBEDDINGS_RC_PATH, 'rb') as f: X_train_bert = pickle.load(f)
        print(f"Loaded: {BERT_TRAIN_EMBEDDINGS_RC_PATH}")
        with open(BERT_TEST_EMBEDDINGS_RC_PATH, 'rb') as f: X_test_bert = pickle.load(f)
        print(f"Loaded: {BERT_TEST_EMBEDDINGS_RC_PATH}")
    except Exception as e:
        print(f"Error loading BERT Recall Class embeddings: {e}. Regenerating...")
        X_train_bert = None
if 'X_train_bert' not in locals() or X_train_bert is None:
    if bert_tokenizer_global is None or bert_model_global is None:
        print("Re-loading BERT model for embedding generation...")
        bert_tokenizer_global = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model_global = TFDistilBertModel.from_pretrained(BERT_MODEL_NAME)
        bert_model_global.trainable = False
    print("Generating BERT embeddings for training set (Recall Class)...")
    X_train_bert = generate_bert_embeddings(X_train_df[TEXT_FEATURE_COLUMN].tolist(), bert_model_global, bert_tokenizer_global, MAX_LEN, BERT_BATCH_SIZE)
    try:
        print(f"Attempting to save: {BERT_TRAIN_EMBEDDINGS_RC_PATH}")
        with open(BERT_TRAIN_EMBEDDINGS_RC_PATH, 'wb') as f: pickle.dump(X_train_bert, f)
        print(f"Successfully saved: {BERT_TRAIN_EMBEDDINGS_RC_PATH}")
    except Exception as e: print(f"Error saving {BERT_TRAIN_EMBEDDINGS_RC_PATH}: {e}")
    print("Generating BERT embeddings for test set (Recall Class)...")
    X_test_bert = generate_bert_embeddings(X_test_df[TEXT_FEATURE_COLUMN].tolist(), bert_model_global, bert_tokenizer_global, MAX_LEN, BERT_BATCH_SIZE)
    try:
        print(f"Attempting to save: {BERT_TEST_EMBEDDINGS_RC_PATH}")
        with open(BERT_TEST_EMBEDDINGS_RC_PATH, 'wb') as f: pickle.dump(X_test_bert, f)
        print(f"Successfully saved: {BERT_TEST_EMBEDDINGS_RC_PATH}")
    except Exception as e: print(f"Error saving {BERT_TEST_EMBEDDINGS_RC_PATH}: {e}")
    if bert_model_global is not None:
        del bert_tokenizer_global, bert_model_global
        gc.collect()

# Preprocess tabular features for NN
X_train_tabular_nn, X_test_tabular_nn, tabular_preprocessor_nn = None, None, None
if os.path.exists(TABULAR_PREPROCESSOR_RC_PATH) and \
   os.path.exists(X_TRAIN_TABULAR_RC_PATH) and \
   os.path.exists(X_TEST_TABULAR_RC_PATH):
    print("Loading preprocessed tabular data for NN Recall Class path...")
    try:
        with open(TABULAR_PREPROCESSOR_RC_PATH, 'rb') as f: tabular_preprocessor_nn = pickle.load(f)
        print(f"Loaded: {TABULAR_PREPROCESSOR_RC_PATH}")
        with open(X_TRAIN_TABULAR_RC_PATH, 'rb') as f: X_train_tabular_nn = pickle.load(f)
        print(f"Loaded: {X_TRAIN_TABULAR_RC_PATH}")
        with open(X_TEST_TABULAR_RC_PATH, 'rb') as f: X_test_tabular_nn = pickle.load(f)
        print(f"Loaded: {X_TEST_TABULAR_RC_PATH}")
    except Exception as e:
        print(f"Error loading tabular Recall Class data for NN: {e}. Regenerating...")
        X_train_tabular_nn = None
if 'X_train_tabular_nn' not in locals() or X_train_tabular_nn is None:
    print("Preprocessing tabular data for NN Recall Class path...")
    # *** ADD DEBUG PRINTS FOR NN TABULAR PREPROCESSOR FEATURE LISTS ***
    print(f"NN Tabular - Input X_train_df columns (before dropping text): {X_train_df.columns.tolist()}")
    print(f"NN Tabular - final_numerical_features: {final_numerical_features}")
    print(f"NN Tabular - final_categorical_features_for_ohe: {final_categorical_features_for_ohe}")
    print(f"NN Tabular - final_pre_one_hot_encoded_features: {final_pre_one_hot_encoded_features}")

    tabular_transformers_list_nn = []
    # Use final lists for ColumnTransformer
    if final_numerical_features:
        tabular_transformers_list_nn.append(('num', StandardScaler(), final_numerical_features))
    if final_categorical_features_for_ohe:
        tabular_transformers_list_nn.append(('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_categorical_features_for_ohe))
    if final_pre_one_hot_encoded_features:
        tabular_transformers_list_nn.append(('passthrough_pre_ohe', 'passthrough', final_pre_one_hot_encoded_features))

    if not tabular_transformers_list_nn:
        print("WARNING: No transformers defined for tabular_preprocessor_nn. X_train_tabular_nn will be empty or error.")
        # Handle case with no tabular features to avoid ColumnTransformer error
        # This might mean X_train_tabular_nn ends up with 0 columns if all lists are empty
        # which would then cause issues with hstack if X_train_bert is not None.
        # A more robust solution might be needed if this case is common.
        X_train_tabular_nn = np.array([]).reshape(len(X_train_df), 0) # Empty array with correct number of rows
        X_test_tabular_nn = np.array([]).reshape(len(X_test_df), 0)
        # Create a dummy preprocessor if needed for saving, or skip saving it
        tabular_preprocessor_nn = 'dummy_no_tabular_features'
    else:
        tabular_preprocessor_nn = ColumnTransformer(
            transformers=tabular_transformers_list_nn,
            remainder='drop'
        )
        X_train_tabular_nn = tabular_preprocessor_nn.fit_transform(X_train_df.drop(columns=[TEXT_FEATURE_COLUMN]))
        X_test_tabular_nn = tabular_preprocessor_nn.transform(X_test_df.drop(columns=[TEXT_FEATURE_COLUMN]))

    print("Saving preprocessed tabular data for NN Recall Class path...")
    try:
        # Only save if it's a real preprocessor
        if not isinstance(tabular_preprocessor_nn, str):
            with open(TABULAR_PREPROCESSOR_RC_PATH, 'wb') as f: pickle.dump(tabular_preprocessor_nn, f)
            print(f"Saved: {TABULAR_PREPROCESSOR_RC_PATH}")
        with open(X_TRAIN_TABULAR_RC_PATH, 'wb') as f: pickle.dump(X_train_tabular_nn, f)
        print(f"Saved: {X_TRAIN_TABULAR_RC_PATH}")
        with open(X_TEST_TABULAR_RC_PATH, 'wb') as f: pickle.dump(X_test_tabular_nn, f)
        print(f"Saved: {X_TEST_TABULAR_RC_PATH}")
    except Exception as e: print(f"Error saving tabular Recall Class data for NN: {e}")

print(f"Shape of X_train_tabular_nn: {X_train_tabular_nn.shape}")
X_train_nn = np.hstack((X_train_tabular_nn, X_train_bert))
X_test_nn = np.hstack((X_test_tabular_nn, X_test_bert))
print(f"Shape of X_train_nn (Tabular + BERT): {X_train_nn.shape}")

# --- 7. Neural Network Model (BERT Extract for Recall Class) ---
print("\n--- Training Neural Network Classifier (BERT Extract) - Recall Class ---")
input_dim = X_train_nn.shape[1]
nn_model_rc = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(64, activation='relu'), Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
adam_optimizer_rc = Adam(learning_rate=1e-4)
nn_model_rc.compile(optimizer=adam_optimizer_rc,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
nn_model_rc.summary()
unique_y_train_labels_rc = np.unique(y_train)
class_weights_sklearn_rc = class_weight.compute_class_weight(
    class_weight='balanced', classes=unique_y_train_labels_rc, y=y_train
)
class_weights_nn_rc = {label: weight for label, weight in zip(unique_y_train_labels_rc, class_weights_sklearn_rc)}
print(f"Class weights for NN (Recall Class): {class_weights_nn_rc}")
history_rc = nn_model_rc.fit(X_train_nn, y_train,
                       epochs=50, batch_size=32, validation_split=0.1,
                       class_weight=class_weights_nn_rc,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
                       verbose=1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_rc.history['loss'], label='Train Loss')
plt.plot(history_rc.history['val_loss'], label='Validation Loss')
plt.title('NN Recall Class Model Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_rc.history['accuracy'], label='Train Accuracy')
plt.plot(history_rc.history['val_accuracy'], label='Validation Accuracy')
plt.title('NN Recall Class Model Accuracy')
plt.legend()
plt.show()
loss_nn_rc, accuracy_nn_rc = nn_model_rc.evaluate(X_test_nn, y_test, verbose=0)
print("\nNeural Network (BERT Extract - Recall Class) - Evaluation:")
print(f"Test Accuracy: {accuracy_nn_rc:.4f}")
print(f"Test Loss: {loss_nn_rc:.4f}")
y_pred_proba_nn_rc = nn_model_rc.predict(X_test_nn)
y_pred_nn_rc = np.argmax(y_pred_proba_nn_rc, axis=1)
print("Classification Report (Recall Class - NN):")
print(classification_report(y_test, y_pred_nn_rc, zero_division=0))
cm_nn_rc = confusion_matrix(y_test, y_pred_nn_rc)
plt.figure(figsize=(8, 6))
if 'target_map' in locals():
    heatmap_labels = sorted(target_map.keys())
else:
    heatmap_labels = sorted(y_test.unique())
sns.heatmap(cm_nn_rc, annot=True, fmt='d', cmap='Blues',
            xticklabels=heatmap_labels, yticklabels=heatmap_labels)
plt.title('NN Recall Class Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# --- 8. Comparison (Recall Class Prediction) ---
print("\n--- Recall Class Prediction Model Comparison ---")
print(f"Random Forest (TF-IDF) Test Accuracy: {accuracy_score(y_test, y_pred_rf_rc):.4f}")
print(f"Neural Network (BERT Extract) Test Accuracy: {accuracy_nn_rc:.4f}")
print("\nThis is a stable baseline for Recall Class prediction. Next steps could involve fine-tuning this NN, or exploring other model tasks.")

# --- Final Save Routine ---
def make_inference_ready():
    """
    Saves all model and preprocessor artifacts needed for production inference.
    """
    print("\n--- Saving Models and Artifacts for Inference ---")

    # Save Random Forest model
    rf_model_path = "rf_model_recall_class.pkl"
    try:
        with open(rf_model_path, "wb") as f:
            pickle.dump(rf_classifier_rc, f)
        print(f"Saved Random Forest model: {rf_model_path}")
    except Exception as e:
        print(f"Error saving RF model: {e}")

    # Save Neural Network model
    nn_model_path = "nn_model_recall_class.h5"
    try:
        nn_model_rc.save(nn_model_path)
        print(f"Saved Neural Network model: {nn_model_path}")
    except Exception as e:
        print(f"Error saving NN model: {e}")
    nn_model_rc.save("nn_model_recall_class.keras")
    print(f"Saved Neural Network model in .keras format: nn_model_recall_class.keras")

    # Save TF-IDF ColumnTransformer (if not already saved above)
    try:
        with open(TFIDF_PREPROCESSOR_RC_PATH, "wb") as f:
            pickle.dump(tfidf_preprocessor, f)
        print(f"Saved TF-IDF preprocessor: {TFIDF_PREPROCESSOR_RC_PATH}")
    except Exception as e:
        print(f"Error saving TF-IDF preprocessor: {e}")

    # Save tabular preprocessor (if it's valid)
    try:
        if not isinstance(tabular_preprocessor_nn, str):
            with open(TABULAR_PREPROCESSOR_RC_PATH, "wb") as f:
                pickle.dump(tabular_preprocessor_nn, f)
            print(f"Saved Tabular preprocessor: {TABULAR_PREPROCESSOR_RC_PATH}")
        else:
            print("Warning: No tabular preprocessor to save (was dummy).")
    except Exception as e:
        print(f"Error saving Tabular preprocessor: {e}")

    print("--- All required inference assets are saved. Ready for deployment. ---")

# Attach at end of training script
make_inference_ready()

Y_TEST_RC_PATH = os.path.join(OUTPUT_DIR_RECALL_CLASS, "y_test_rc.pkl")

with open(Y_TEST_RC_PATH, 'wb') as f:
    pickle.dump(y_test, f)
print(f"Saved: {Y_TEST_RC_PATH}")
