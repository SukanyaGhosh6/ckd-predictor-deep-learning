import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Rename target column for clarity
    df.rename(columns={'classification': 'target'}, inplace=True)

    # Encode categorical columns
    label_cols = df.select_dtypes(include=['object']).columns
    for col in label_cols:
        df[col] = df[col].astype(str).str.strip().replace('?', pd.NA)
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = LabelEncoder().fit_transform(df[col])

    # Convert numeric columns and fill missing values
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train-test split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, df.columns[:-1]



