from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_model_stochastic_gradient_descent_regressor_sgd_regressor(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['Timestamp', 'Age', 'Gender', 'Branch', 'Registration Type'], inplace=True)

    X = df.drop('University GPA', axis=1)
    y = df['University GPA']

    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    sgd_model.fit(X_train, y_train)

    y_pred = sgd_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
