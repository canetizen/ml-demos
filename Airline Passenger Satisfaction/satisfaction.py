import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to handle outliers
def handle_outliers(df, numeric_columns):
    def find_outliers_IQR(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        return outliers

    outliers_indices = set()
    for column in numeric_columns:
        outliers_indices.update(find_outliers_IQR(df, column))
    
    df_no_outliers = df.drop(index=outliers_indices)
    return df_no_outliers

# Functions for encoding and scaling initialization and application
def initialize_encoders(df, categorical_columns):
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        le.fit(df[column].astype(str))
        label_encoders[column] = le
    return label_encoders

def apply_encoders(df, categorical_columns, label_encoders):
    for column in categorical_columns:
        le = label_encoders[column]
        df[column] = le.transform(df[column].astype(str))
    return df

def initialize_scaler(df, numeric_columns):
    scaler = RobustScaler()
    scaler.fit(df[numeric_columns])
    return scaler

def apply_scaler(df, numeric_columns, scaler):
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df

# Function to impute missing values
def impute_missing_values(df, categorical_columns, numeric_columns):
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    return df

# Preprocessing function
def preprocess_data(df, label_encoders=None, scaler=None, is_train=True):
    # Drop unnecessary columns
    df = df.drop(["Unnamed: 0", "id"], axis=1, errors='ignore')

    # Identify categorical and numerical columns
    threshold = 10
    categorical_columns = [col for col in df.columns if df[col].nunique() < threshold and col != "satisfaction"]
    numeric_columns = [col for col in df.columns if col not in categorical_columns + ["satisfaction"]]

    # Handle outliers only if it is training data
    if is_train:
        df = handle_outliers(df, numeric_columns)

    # Encode categorical variables
    if is_train:
        label_encoders = initialize_encoders(df, categorical_columns)
    df = apply_encoders(df, categorical_columns, label_encoders)

    # Impute missing values
    df = impute_missing_values(df, categorical_columns, numeric_columns)

    # Scale numerical variables
    if is_train:
        scaler = initialize_scaler(df, numeric_columns)
    df = apply_scaler(df, numeric_columns, scaler)

    # Encode the target variable to 0 and 1
    target = None
    if 'satisfaction' in df.columns:
        target = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
        df = df.drop(['satisfaction'], axis=1)

    return df, target, label_encoders, scaler

# Evaluation function
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

# Load the datasets
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')

# Preprocess the training data
X_train, y_train, label_encoders, scaler = preprocess_data(df_train, is_train=True)

# Preprocess the test data using the same encoders and scaler
X_test, y_test, _, _ = preprocess_data(df_test, label_encoders=label_encoders, scaler=scaler, is_train=False)

# Train the logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)
