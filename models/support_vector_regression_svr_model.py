# إسقاط الأعمدة غير الضرورية
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def train_model_support_vector_regression_svr_model(file_path):
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
    df.drop(columns=['Timestamp', 'Age', 'Gender', 'Branch', 'Registration Type'], inplace=True)

    # تحديد المتغيرات المستقلة والتابعة
    X = df.drop('University GPA', axis=1)
    y = df['University GPA']

    # إذا كانت بعض الأعمدة تحتوي على قيم نصية، نحتاج لترميزها
    for col in X.columns:
        if X[col].dtype == 'object':
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])

    # تطبيع البيانات باستخدام StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # إنشاء وتدريب نموذج SVR
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train.ravel())

    # التنبؤ بالقيم على مجموعة الاختبار
    y_pred = svr.predict(X_test)

    # تقييم النموذج
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # متوسط مربع الخطأ
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)