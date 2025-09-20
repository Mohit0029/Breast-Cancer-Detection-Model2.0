import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import time

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*60)
    print(f"| {title.upper():^56} |")
    print("="*60)

def print_step(message):
    """Prints a formatted step message."""
    print(f"[INFO] {message}")
    time.sleep(0.5)

# --- SCRIPT START ---
print_header("Breast Cancer Model Training Pipeline")

# --- 1. Load Dataset ---
print_step("Connecting to data source and loading dataset...")
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_names = [
        "sample_code_number", "clump_thickness", "uniform_cell_size",
        "uniform_cell_shape", "marginal_adhesion", "single_epithelial_size",
        "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"
    ]
    data = pd.read_csv(url, names=column_names)
    print_step("Dataset loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit()

# --- 2. Preprocessing Data ---
print_header("Data Preprocessing")
print_step("Handling missing values ('?')...")
initial_rows = len(data)
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
cleaned_rows = len(data)
print_step(f"Removed {initial_rows - cleaned_rows} rows with missing data.")
print_step("Converting 'bare_nuclei' column to integer type...")
data['bare_nuclei'] = data['bare_nuclei'].astype(int)
print_step("Preprocessing complete.")

# --- 3. Feature and Target Definition ---
print_header("Feature and Target Definition")
feature_columns = [
    "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
    "marginal_adhesion", "single_epithelial_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses"
]
X = data[feature_columns]
y = data["class"]  # Target: 2 for benign, 4 for malignant
print_step("Features (X) and Target (y) have been defined.")

# --- 4. Model Training ---
print_header("Model Training")
print_step("Splitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print_step("Initializing RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
print_step("Training the model... (This may take a moment)")
model.fit(X_train, y_train)
print_step("Model training completed successfully.")

# --- 5. Evaluation ---
print_header("Model Evaluation")
accuracy = model.score(X_test, y_test)
print_step("Calculating model accuracy on the unseen test set...")
print("-" * 60)
print(f"| {'Final Model Accuracy':<40} | {accuracy * 100:>12.2f}% |")
print("-" * 60)


# --- 6. Save Model ---
print_header("Saving Model")
print_step("Serializing the trained model to 'model.pkl'...")
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
print_step("Model successfully saved as model.pkl.")

print_header("Training Pipeline Finished")

