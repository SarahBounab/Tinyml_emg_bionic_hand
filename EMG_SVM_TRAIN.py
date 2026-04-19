import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Parameters 
WINDOW_DURATION = 0.3     # seconds per window
STEP_RATIO      = 0.75      
STEP_DURATION   = WINDOW_DURATION * STEP_RATIO
PURITY_THRESH   = 0.90     


#data
df = pd.read_csv('concatenated_9v(FilteredV)_class01.csv') #change file name

# assume your CSV has a 'time' column in seconds
times  = df['Time'].values
data   = df['FilteredV'].values
labels = df['Label'].values

# Feature extraction
def extract_features(segment: np.ndarray) -> list:
    return [
        np.min(segment),
        np.max(segment),
        # np.mean(segment),
        np.std(segment),
        np.mean(np.abs(segment)),
        # ((segment[:-1] * segment[1:]) < 0).sum(),
        # np.median(segment),
        # np.ptp(segment),
        # np.sum(segment**2),
        # scipy.stats.kurtosis(segment),
        # np.max(segment) - np.min(segment),
        # np.sqrt(np.mean(segment**2)),
    ]


X, y = [], []
total_windows = 0

t_start = times.min()
t_end   = times.max()

while t_start + WINDOW_DURATION <= t_end:
    mask = (times >= t_start) & (times < t_start + WINDOW_DURATION)
    seg = data[mask]
    lbls = labels[mask]
    if len(lbls) == 0:
        t_start += STEP_DURATION
        continue

    total_windows += 1
    # majority vote label and purity
    counts = Counter(lbls)
    label_most, count = counts.most_common(1)[0]
    purity = count / len(lbls)

    if purity >= PURITY_THRESH:
     X.append(extract_features(seg))
     y.append(label_most)

    t_start += STEP_DURATION

print(f"Total windows scanned: {total_windows}")
print(f"Windows kept (≥{PURITY_THRESH*100:.0f}% purity): {len(X)}")


X = np.array(X)
y = np.array(y)


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

feature_names = [
    'Min', 'Max', 'Mean', 'Std', 'MeanAbs', 'ZeroCross', 'Median',
    'PeakToPeak', 'SumSquares', 'Kurtosis', 'Range', 'RMS'
]


# Boxplot 
plt.figure(figsize=(12, 6))
for i in range(X.shape[1]):
    plt.subplot(3, 4, i + 1)  # This arranges 9 plots in a 3x3 grid
    sns.boxplot(x=y_train, y=X_scaled[:, i], showfliers=False)


    plt.title(f"Boxplot of {feature_names[i]}")
    plt.xlabel("Class")
    plt.ylabel("Feature Value")

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()



pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svc',    SVC(kernel='linear'))
])

param_grid = {
    'svc__C': [0.01,0.1, 1, 10, 100]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_['svc__C'])
print("Validation accuracy:", grid.score(X_valid, y_valid))
best_model = grid.best_estimator_


joblib.dump(grid.best_estimator_, 'emg_svc_pipeline_time_based2_t.joblib')
print("Saved pipeline to 'emg_svc_pipeline_time_based.joblib'")


# Evaluation
y_pred = best_model.predict(X_valid)
print("\nClassification Report:\n", classification_report(y_valid, y_pred))
print("Accuracy Score:", accuracy_score(y_valid, y_pred))

# Confusion Matrix 
conf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matrice de Confusion SVM linéaire threshold 90% ")
plt.xlabel("Label prédit ")
plt.ylabel("Vrai Label")
plt.tight_layout()
plt.show()



#  Extract Model Parameters 
svc_model = best_model.named_steps['svc']
scaler_model = best_model.named_steps['scaler']

print("\n=== SVM Parameters ===")
print("Weights (w):", svc_model.coef_)
print("Bias (b):", svc_model.intercept_)

print("\n=== Scaler Parameters ===")
print("Feature Min:", scaler_model.data_min_)
print("Feature Max:", scaler_model.data_max_)
print("Feature Range:", scaler_model.data_range_)
