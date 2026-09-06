"""
Weather Prediction Model Training Script
Generates a realistic dataset, trains a DecisionTreeClassifier,
evaluates metrics, and saves the model and preprocessor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

# Resolve paths relative to this file so training works from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# STEP 1: Generate Realistic Weather Dataset
# ============================================================
print("Generating realistic weather dataset...")

n_samples = 5000

# Define weather classes
weather_classes = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Thunderstorm', 'Foggy']

# Generate data with realistic correlations
data = []
for i in range(n_samples):
    weather_class = np.random.choice(weather_classes)

    if weather_class == 'Sunny':
        temperature = np.random.normal(30, 5)
        humidity = np.random.normal(35, 10)
        wind_speed = np.random.normal(10, 4)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(10, 8)
        pressure = np.random.normal(1018, 5)
        visibility = np.random.normal(15, 3)
        precipitation = np.random.normal(0, 0.5)
        uv_index = np.random.normal(8, 2)
        dew_point = np.random.normal(12, 4)

    elif weather_class == 'Partly Cloudy':
        temperature = np.random.normal(25, 4)
        humidity = np.random.normal(50, 10)
        wind_speed = np.random.normal(15, 5)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(40, 12)
        pressure = np.random.normal(1015, 5)
        visibility = np.random.normal(12, 3)
        precipitation = np.random.normal(1, 1)
        uv_index = np.random.normal(5, 1.5)
        dew_point = np.random.normal(14, 4)

    elif weather_class == 'Cloudy':
        temperature = np.random.normal(18, 4)
        humidity = np.random.normal(65, 10)
        wind_speed = np.random.normal(18, 6)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(75, 12)
        pressure = np.random.normal(1010, 6)
        visibility = np.random.normal(8, 3)
        precipitation = np.random.normal(2, 1.5)
        uv_index = np.random.normal(3, 1)
        dew_point = np.random.normal(13, 3)

    elif weather_class == 'Rainy':
        temperature = np.random.normal(15, 4)
        humidity = np.random.normal(80, 8)
        wind_speed = np.random.normal(25, 8)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(90, 8)
        pressure = np.random.normal(1005, 7)
        visibility = np.random.normal(6, 2)
        precipitation = np.random.normal(8, 4)
        uv_index = np.random.normal(2, 1)
        dew_point = np.random.normal(12, 3)

    elif weather_class == 'Thunderstorm':
        temperature = np.random.normal(22, 5)
        humidity = np.random.normal(85, 7)
        wind_speed = np.random.normal(45, 15)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(95, 5)
        pressure = np.random.normal(998, 8)
        visibility = np.random.normal(4, 2)
        precipitation = np.random.normal(15, 7)
        uv_index = np.random.normal(1.5, 0.8)
        dew_point = np.random.normal(18, 4)

    elif weather_class == 'Foggy':
        temperature = np.random.normal(10, 3)
        humidity = np.random.normal(92, 5)
        wind_speed = np.random.normal(5, 2)
        wind_direction = np.random.uniform(0, 360)
        cloud_cover = np.random.normal(70, 15)
        pressure = np.random.normal(1012, 5)
        visibility = np.random.normal(2, 1)
        precipitation = np.random.normal(0.5, 0.5)
        uv_index = np.random.normal(2, 1)
        dew_point = np.random.normal(9, 3)

    data.append({
        'temperature': round(temperature, 1),
        'humidity': round(humidity, 1),
        'wind_speed': round(wind_speed, 1),
        'wind_direction': round(wind_direction, 1),
        'cloud_cover': round(cloud_cover, 1),
        'pressure': round(pressure, 1),
        'visibility': round(visibility, 1),
        'precipitation': round(precipitation, 1),
        'uv_index': round(uv_index, 1),
        'dew_point': round(dew_point, 1),
        'weather': weather_class
    })

df = pd.DataFrame(data)

# ============================================================
# STEP 2: Clean Data - Handle Missing Values & Duplicates
# ============================================================
print(f"Dataset shape before cleaning: {df.shape}")

# Introduce some realistic missing values (then handle them)
missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
df.loc[missing_indices[:20], 'temperature'] = np.nan
df.loc[missing_indices[20:40], 'humidity'] = np.nan
df.loc[missing_indices[40:60], 'wind_speed'] = np.nan
df.loc[missing_indices[60:80], 'visibility'] = np.nan
df.loc[missing_indices[80:100], 'uv_index'] = np.nan

# Handle missing values - fill with median for numerical columns
numerical_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                  'cloud_cover', 'pressure', 'visibility', 'precipitation',
                  'uv_index', 'dew_point']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df = df.drop_duplicates()

# Clip values to realistic ranges
df['humidity'] = df['humidity'].clip(5, 100)
df['wind_speed'] = df['wind_speed'].clip(0, 120)
df['wind_direction'] = df['wind_direction'].clip(0, 360)
df['cloud_cover'] = df['cloud_cover'].clip(0, 100)
df['pressure'] = df['pressure'].clip(950, 1050)
df['visibility'] = df['visibility'].clip(0, 30)
df['precipitation'] = df['precipitation'].clip(0, 50)
df['uv_index'] = df['uv_index'].clip(0, 14)
df['temperature'] = df['temperature'].clip(-20, 50)
df['dew_point'] = df['dew_point'].clip(-20, 40)

# ============================================================
# STEP 3: Save Dataset
# ============================================================
df.to_csv(os.path.join(DATA_DIR, 'weather_data.csv'), index=False)
print(f"Dataset saved to data/weather_data.csv")
print(f"Class distribution:\n{df['weather'].value_counts()}")

# ============================================================
# STEP 4: Prepare Features and Target
# ============================================================
feature_columns = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                   'cloud_cover', 'pressure', 'visibility', 'precipitation',
                   'uv_index', 'dew_point']

X = df[feature_columns].values
y = df['weather'].values

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# ============================================================
# STEP 5: Train/Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================
# STEP 6: Train Calibrated Decision Tree Classifier
# ============================================================
print("\nTraining Calibrated Decision Tree Classifier...")
# Wrap DecisionTreeClassifier with CalibratedClassifierCV to produce
# well-calibrated probabilities instead of the extreme 0%/100% outputs
# that pure leaf nodes of a deep tree produce.
base_tree = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
dt_classifier = CalibratedClassifierCV(base_tree, method='sigmoid', cv=5)
dt_classifier.fit(X_train, y_train)
print("Model training complete!")

# ============================================================
# STEP 7: Evaluate Model
# ============================================================
y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*50)

# ============================================================
# STEP 8: Save Model and Metrics
# ============================================================
joblib.dump(dt_classifier, os.path.join(MODEL_DIR, 'weather_model.pkl'))
print("\nModel saved to model/weather_model.pkl")

# Save metrics
metrics = {
    'accuracy': round(accuracy * 100, 2),
    'precision': round(precision * 100, 2),
    'recall': round(recall * 100, 2),
    'f1_score': round(f1 * 100, 2),
    'total_records': len(df),
    'train_records': len(X_train),
    'test_records': len(X_test),
    'num_features': len(feature_columns),
    'num_classes': len(weather_classes),
    'classes': weather_classes,
    'feature_names': feature_columns
}
joblib.dump(metrics, os.path.join(MODEL_DIR, 'metrics.pkl'))
print("Metrics saved to model/metrics.pkl")

print("\nTraining pipeline complete!")


print(f"Dataset shape after cleaning: {df.shape}")

