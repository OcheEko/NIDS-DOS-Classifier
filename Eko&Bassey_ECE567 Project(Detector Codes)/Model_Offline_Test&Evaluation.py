import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
 

# Load your dataset
data = pd.read_csv("/Users/ochee/Downloads/files_proj/files/flows_benign_and_DoS.csv")

# Remove extra spaces from column names
data.columns = [column.strip() for column in data.columns]

# Select the columns you want to use for training
selected_columns = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Packets/s',
    'Flow IAT Mean', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Init_Win_bytes_backward'
]

# Extract the features (X) and labels (y)
X = data[selected_columns]
y = data.iloc[:, -1]  # Assuming the label is in the last column

# Replace large values with NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Use RobustScaler for feature scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


########################################################################################################

#Load your Attack data set
data = pd.read_csv("/Users/ochee/Downloads/files_proj/files/output.csv")


# # Extract the features (X) and labels (y)
X_1 = data[selected_columns]
y_1 = data.iloc[:, -1]  # Assuming the label is in the last column

# Replace large values with NaNs
X_1.replace([np.inf, -np.inf], np.nan, inplace=True)

# Use RobustScaler for feature scaling
scaler = RobustScaler()
X_scaled_1 = scaler.fit_transform(X_1)

# Split the data into training and testing sets
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_scaled_1, y_1, test_size=0.3, random_state=42)

# Train the model
model.fit(X_train_1, y_train_1)

# Make predictions
y_pred_1 = model.predict(X_test_1)

# Classification Report
print("Classification Report:")
print(classification_report(y_test_1, y_pred_1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_1, y_pred_1)
print("Confusion Matrix:")
print(conf_matrix)

#Calculate ROC AUC
roc_auc = roc_auc_score(y_test_1, y_pred_1)
print("ROC AUC:", roc_auc)

# Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate False Positive Rate (FPR)
FPR = FP / (FP + TN)
print("False Positive Rate (FPR):", FPR)

# Calculate Detection Rate (DR) or True Positive Rate (TPR)
DR = TP / (TP + FN)
print("Detection Rate (DR):", DR)

# Evaluate the model
accuracy = accuracy_score(y_test_1, y_pred_1)
print("Accuracy:", accuracy)




