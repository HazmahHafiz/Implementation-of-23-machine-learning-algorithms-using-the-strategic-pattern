from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import os
from models.random_forest_regressor import train_model_random_forest_regressor
from models.adaboost_regressor import train_model_adaboost_regressor
from models.bayesian_ridge_regression import train_model_bayesian_ridge_regressor
from models.catboost_regressor import train_model_catboost_regressor
from models.decision_tree_regressor import train_model_decision_tree_regressor
from models.elastic_net_regression import train_model_elastic_net_regressor
from models.extra_trees_regressor import train_model_extra_trees_regressor
from models.gaussian_process_regression_gpr import train_model_gaussian_process_regressor
from models.gradient_boosting_regressors import train_model_gradient_boosting_regressor
from models.histogram_based_gradient_boosting_regressor_histgradientboostingregressor import train_model_histogram_based_gradient_boosting_regressor
from models.k_nearest_neighbors_regressor_knn import train_model_k_nearest_neighbors_regressor_knn
from models.lasso_regression import train_model_lasso_regressor
from models.lightgbm_regressor import train_model_lightgbm_regressor
from models.linear_regression_model import train_model_linear_regression_model
from models.partial_least_squares_regression_plsr import train_model_partial_least_squares_regression_model
from models.polynomial_regression_model import train_model_polynomial_regression_model
from models.principal_component_regression_pcr import train_model_principal_component_regression_pcr
from models.quantile_regression import train_model_quantile_regression
from models.ridge_regression import train_model_ridge_regression
from models.robust_regression import train_model_robust_regression
from models.stochastic_gradient_descent_regressor_sgd_regressor import train_model_stochastic_gradient_descent_regressor_sgd_regressor
from models.support_vector_regression_svr_model import train_model_support_vector_regression_svr_model









def create_folders(path):
    main = os.path.join(path, "Data charts")
    os.makedirs(main, exist_ok=True)
    os.makedirs(os.path.join(path, "After data cleaning process"), exist_ok=True)
    
    for sub in ["after clean", "after trimmed", "before"]:
        os.makedirs(os.path.join(main, sub), exist_ok=True)

create_folders("./")
# قراءة البيانات وإعادة تسمية الأعمدة
df = pd.read_csv('Data.csv', encoding='utf-8', sep=',')
column_rename = {
    'طابع زمني': 'Timestamp',
    'العمر': 'Age', 
    'الجنس': 'Gender',
    'الفرع': 'Branch',
    'هل انت تجسير ام نقل ام تسجيل اساسي': 'Registration Type',
    'معدل الشهادة الثانوية': 'High School GPA',
    'نسبة الحضور الاسبوعية من مئة': 'Weekly Attendance Percentage',
    'متوسط عدد ساعات الدراسة الاسبوعية': 'Weekly Study Hours',
    'متوسط عدد ساعات الدراسة اليومية': 'Daily Study Hours',
    'هل انت متزوج': 'Marital Status',
    'هل تعمل': 'Employment Status',
    'هل لديك جدول دراسي منظم': 'Has Study Schedule',
    'معدلك التراكمي (هذا السؤال مهم جدا جدا)': 'University GPA'
}
df.rename(columns=column_rename, inplace=True)
print(df.head())
print("Number of rows before removing outliers:", len(df))
# عملية تنظيف البيانات
df.drop(df[~df['High School GPA'].between(50, 100)].index, inplace=True)
df.to_csv('./After data cleaning process/1_Data_cleaning_cleaned_GPA.csv', index=False, encoding='utf-8-sig')
df.drop(df[~df['Weekly Attendance Percentage'].between(0, 100)].index, inplace=True)
df.to_csv('./After data cleaning process/2_Data_cleaning_cleaned_attendance.csv', index=False, encoding='utf-8-sig')
df.drop(df[~df['Weekly Study Hours'].between(1, 84)].index, inplace=True)
df.to_csv('./After data cleaning process/3_Data_cleaning_cleaned_weekly_hours.csv', index=False, encoding='utf-8-sig')
df.drop(df[~df['Daily Study Hours'].between(1, 12)].index, inplace=True)
df.to_csv('./After data cleaning process/4_Data_cleaning_cleaned_daily_hours.csv', index=False, encoding='utf-8-sig')
le = LabelEncoder()
df['Marital Status'] = le.fit_transform(df['Marital Status'])
df['Employment Status'] = le.fit_transform(df['Employment Status'])
df['Has Study Schedule'] = le.fit_transform(df['Has Study Schedule'])
df.to_csv('./After data cleaning process/5_Data_cleaning_encoded.csv', index=False, encoding='utf-8-sig')
df.loc[df['University GPA'] > 4, 'University GPA'] = (df.loc[df['University GPA'] > 4, 'University GPA'] / 100) * 4
df.drop(df[(df['University GPA'] > 4) | (df['University GPA'] < 2.4)].index, inplace=True)
df.to_csv('./After data cleaning process/6_Data_cleaning_cleaned_university_gpa.csv', index=False, encoding='utf-8-sig')

# تقليم البيانات
df = pd.read_csv("./After data cleaning process/6_Data_cleaning_cleaned_university_gpa.csv", encoding='utf-8', sep=',')
df.rename(columns=column_rename, inplace=True)
df = df.sort_values(by=['University GPA', 'Weekly Study Hours', 'Daily Study Hours'], ascending=False)
numeric_cols = ['High School GPA', 'Weekly Attendance Percentage', 'Weekly Study Hours', 'Daily Study Hours', 'University GPA']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print("Number of rows after removing outliers:", len(df))
df.to_csv('data_trimmed.csv', index=False,  encoding='utf-8-sig')

# رسم المخططات
df = pd.read_csv('data.csv', encoding='utf-8', sep=',')
df.rename(columns=column_rename, inplace=True)
plt.style.use('seaborn-v0_8')
# العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.plot(df['Weekly Study Hours'], df['University GPA'], marker='o')
plt.title('Relationship between Weekly Study Hours and University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/before/1_العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مقارنة معدل الجامعة حسب نسبة الحضور
plt.figure(figsize=(10, 6))
avg_gpa_by_attendance = df.groupby('Weekly Attendance Percentage')['University GPA'].mean()
plt.bar(avg_gpa_by_attendance.index, avg_gpa_by_attendance.values, color='blue', width=10)
plt.title('Average GPA by Weekly Attendance Percentage')
plt.xlabel('Weekly Attendance Percentage')
plt.ylabel('Average GPA')
plt.savefig('Data charts/before/2_مقارنة معدل الجامعة حسب نسبة الحضور.png')
plt.close()
#توزيع ساعات الدراسة الأسبوعية واليومية
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Weekly Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Weekly Study Hours')
plt.xlabel('Weekly Study Hours') 
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(df['Daily Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Daily Study Hours')
plt.xlabel('Daily Study Hours')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Data charts/before/3_توزيع ساعات الدراسة الأسبوعية واليومية.png')
plt.close()
# مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.scatter(df['Weekly Study Hours'], df['University GPA'], alpha=0.5)
plt.title('Scatter Plot of Weekly Study Hours vs University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/before/4_مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مخطط الصندوق لساعات الدراسة الأسبوعية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Weekly Study Hours'], widths=0.7)
plt.title('Box Plot of Weekly Study Hours Distribution', fontsize=14)
plt.ylabel('Weekly Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/before/5_مخطط الصندوق لساعات الدراسة الأسبوعية.png')
plt.close()
# مخطط الصندوق لساعات الدراسة اليومية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Daily Study Hours'], widths=0.7)
plt.title('Box Plot of Daily Study Hours Distribution', fontsize=14)
plt.ylabel('Daily Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/before/6_مخطط الصندوق لساعات الدراسة اليومية.png')
plt.close()
df = pd.read_csv("./After data cleaning process/6_Data_cleaning_cleaned_university_gpa.csv", encoding='utf-8', sep=',')
df.rename(columns=column_rename, inplace=True)
plt.style.use('seaborn-v0_8')
# العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.plot(df['Weekly Study Hours'], df['University GPA'], marker='o')
plt.title('Relationship between Weekly Study Hours and University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/after clean/1_العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مقارنة معدل الجامعة حسب نسبة الحضور
plt.figure(figsize=(10, 6))
avg_gpa_by_attendance = df.groupby('Weekly Attendance Percentage')['University GPA'].mean()
plt.bar(avg_gpa_by_attendance.index, avg_gpa_by_attendance.values, color='blue', width=10)
plt.title('Average GPA by Weekly Attendance Percentage')
plt.xlabel('Weekly Attendance Percentage')
plt.ylabel('Average GPA')
plt.savefig('Data charts/after clean/2_مقارنة معدل الجامعة حسب نسبة الحضور.png')
plt.close()
#توزيع ساعات الدراسة الأسبوعية واليومية
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Weekly Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Weekly Study Hours')
plt.xlabel('Weekly Study Hours') 
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(df['Daily Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Daily Study Hours')
plt.xlabel('Daily Study Hours')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Data charts/after clean/3_توزيع ساعات الدراسة الأسبوعية واليومية.png')
plt.close()
# مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.scatter(df['Weekly Study Hours'], df['University GPA'], alpha=0.5)
plt.title('Scatter Plot of Weekly Study Hours vs University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/after clean/4_مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مخطط الصندوق لساعات الدراسة الأسبوعية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Weekly Study Hours'], widths=0.7)
plt.title('Box Plot of Weekly Study Hours Distribution', fontsize=14)
plt.ylabel('Weekly Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/after clean/5_مخطط الصندوق لساعات الدراسة الأسبوعية.png')
plt.close()
# مخطط الصندوق لساعات الدراسة اليومية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Daily Study Hours'], widths=0.7)
plt.title('Box Plot of Daily Study Hours Distribution', fontsize=14)
plt.ylabel('Daily Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/after clean/6_مخطط الصندوق لساعات الدراسة اليومية.png')
plt.close()

df = pd.read_csv("data_trimmed.csv", encoding='utf-8', sep=',')
df.rename(columns=column_rename, inplace=True)
plt.style.use('seaborn-v0_8')
# العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.plot(df['Weekly Study Hours'], df['University GPA'], marker='o')
plt.title('Relationship between Weekly Study Hours and University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/after trimmed/1_العلاقة بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مقارنة معدل الجامعة حسب نسبة الحضور
plt.figure(figsize=(10, 6))
avg_gpa_by_attendance = df.groupby('Weekly Attendance Percentage')['University GPA'].mean()
plt.bar(avg_gpa_by_attendance.index, avg_gpa_by_attendance.values, color='blue', width=10)
plt.title('Average GPA by Weekly Attendance Percentage')
plt.xlabel('Weekly Attendance Percentage')
plt.ylabel('Average GPA')
plt.savefig('Data charts/after trimmed/2_مقارنة معدل الجامعة حسب نسبة الحضور.png')
plt.close()
#توزيع ساعات الدراسة الأسبوعية واليومية
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Weekly Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Weekly Study Hours')
plt.xlabel('Weekly Study Hours') 
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(df['Daily Study Hours'], bins=20, edgecolor='black')
plt.title('Distribution of Daily Study Hours')
plt.xlabel('Daily Study Hours')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Data charts/after trimmed/3_توزيع ساعات الدراسة الأسبوعية واليومية.png')
plt.close()
# مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي
plt.figure(figsize=(10, 6))
plt.scatter(df['Weekly Study Hours'], df['University GPA'], alpha=0.5)
plt.title('Scatter Plot of Weekly Study Hours vs University GPA')
plt.xlabel('Weekly Study Hours')
plt.ylabel('University GPA')
plt.grid(True)
plt.savefig('Data charts/after trimmed/4_مخطط التشتت بين ساعات الدراسة الأسبوعية والمعدل التراكمي.png')
plt.close()
# مخطط الصندوق لساعات الدراسة الأسبوعية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Weekly Study Hours'], widths=0.7)
plt.title('Box Plot of Weekly Study Hours Distribution', fontsize=14)
plt.ylabel('Weekly Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/after trimmed/5_مخطط الصندوق لساعات الدراسة الأسبوعية.png')
plt.close()
# مخطط الصندوق لساعات الدراسة اليومية
plt.figure(figsize=(10, 6))
plt.boxplot(df['Daily Study Hours'], widths=0.7)
plt.title('Box Plot of Daily Study Hours Distribution', fontsize=14)
plt.ylabel('Daily Study Hours', fontsize=12)
plt.grid(True)
plt.savefig('Data charts/after trimmed/6_مخطط الصندوق لساعات الدراسة اليومية.png')
plt.close()
file="data_trimmed.csv"
print("-----------------------Training Random Forest Regressor--------------------")
train_model_random_forest_regressor(file)
print("-----------------------Training Adaboost Regressor------------------------")
train_model_adaboost_regressor(file)
print("-----------------------Training Bayesian Ridge Regressor-------------------")
train_model_bayesian_ridge_regressor(file)
print("-----------------------Training Catboost Regressor-------------------------")
train_model_catboost_regressor(file)
print("-----------------------Training Decision Tree Regressor--------------------")
train_model_decision_tree_regressor(file)
print("-----------------------Training Elastic Net Regressor------------------------")
train_model_elastic_net_regressor(file)
print("-----------------------Training Extra Trees Regressor------------------------")
train_model_extra_trees_regressor(file)
print("-----------------------Training Gaussian Process Regressor-------------------")
train_model_gaussian_process_regressor(file)
print("-----------------------Training Gradient Boosting Regressor------------------------")
train_model_gradient_boosting_regressor(file)
print("-----------------------Training Histogram Based Gradient Boosting Regressor------------------------")
train_model_histogram_based_gradient_boosting_regressor(file)
print("-----------------------Training K Nearest Neighbors Regressor------------------------")
train_model_k_nearest_neighbors_regressor_knn(file)
print("-----------------------Training Lasso Regressor------------------------")
train_model_lasso_regressor(file)
print("-----------------------Training LightGBM Regressor------------------------")
train_model_lightgbm_regressor(file)
print("-----------------------Training Linear Regression Model------------------------")
train_model_linear_regression_model(file)
print("-----------------------Training Partial Least Squares Regression Model-------------------")
train_model_partial_least_squares_regression_model(file)
print("-----------------------Training Polynomial Regression Model------------------------")
train_model_polynomial_regression_model(file)
print("-----------------------Training Principal Component Regression Model-------------------")
train_model_principal_component_regression_pcr(file)
print("-----------------------Training Quantile Regression Model------------------------")
train_model_quantile_regression(file)
print("-----------------------Training Ridge Regression Model------------------------")
train_model_ridge_regression(file)
print("-----------------------Training Robust Regression Model------------------------")
train_model_robust_regression(file)
print("-----------------------Training Stochastic Gradient Descent Regressor------------------------")
train_model_stochastic_gradient_descent_regressor_sgd_regressor(file)
print("-----------------------Training Support Vector Regression Model------------------------")
train_model_support_vector_regression_svr_model(file)

