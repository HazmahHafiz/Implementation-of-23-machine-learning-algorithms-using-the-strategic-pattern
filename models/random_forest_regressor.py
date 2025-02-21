from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_model_random_forest_regressor(file_path):
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    df.drop(columns=['Timestamp', 'Age', 'Gender', 'Branch', 'Registration Type'], inplace=True)

    X = df.drop('University GPA', axis=1)
    y = df['University GPA']

    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    return rf_regressor
