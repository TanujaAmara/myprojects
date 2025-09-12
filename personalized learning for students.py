#1. Data Collection 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.metrics import r2_score, mean_squared_error, classification_report, 
accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
# Load dataset 
df = pd.read_csv("student_data1.csv") 
print(df.head()) 

#2. Exploratory Data Analysis (EDA) 
df.head(10) 
df.tail() 
df.info() 
df.describe(include="0") 
print("Duplicate Records:", df.duplicated().sum()) 
print("Missing Values:\n", df.isnull().sum()) 
 
# New Columns: Avg Score & Performance Category 
df['Avg Score'] = ((df['G1'] + df['G2'] + df['G3']) / 3) 
df['Performance Category'] = pd.qcut(df['Avg Score'], q=3, labels=["Low", "Medium", 
"High"]) 
 
# Pivot tables 
print(df.pivot_table(index='Mjob', values='Avg Score', aggfunc='mean')) 
print(df.pivot_table(index='Fjob', values='Avg Score', aggfunc='mean')) 
print(df.pivot_table(index='Pstatus', values='Avg Score', aggfunc='mean')) 
print(df.pivot_table(index='school', values='failures', aggfunc='mean')) 
print(df.pivot_table(index='school', values='Avg Score', aggfunc='mean')) 
 
# Value Counts 
print(df['Mjob'].value_counts()) 
print(df['Fjob'].value_counts()) 
print(df['school'].value_counts()) 
 
#3. Data Preprocessing + Feature Extraction 
# Label Encoding 
df_encoded = df.copy() 
label_enc = LabelEncoder() 
for col in df_encoded.select_dtypes(include='object').columns: 
    df_encoded[col] = label_enc.fit_transform(df_encoded[col]) 
 
# Add pass/fail classification 
df_encoded['pass'] = df_encoded['G3'].apply(lambda x: 1 if x >= 10 else 0) 
 
#4. Implement ML Algorithms (Hyperparameter Tuning + Cross-Validation) 
#Regression 
 
 
X_reg = df_encoded.drop(columns=['G3', 'pass']) 
y_reg = df_encoded['G3'] 
 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, 
test_size=0.2, random_state=42) 
 
param_grid_reg = { 
    'n_estimators': [50, 100, 150], 
    'max_depth': [None, 10, 20], 
    'min_samples_split': [2, 5] 
} 
 
rf_reg = RandomForestRegressor(random_state=42) 
grid_reg = GridSearchCV(rf_reg, param_grid=param_grid_reg, cv=5, scoring='r2', 
n_jobs=-1) 
grid_reg.fit(X_train_reg, y_train_reg) 
 
best_rf_reg = grid_reg.best_estimator_ 
y_pred_reg = best_rf_reg.predict(X_test_reg) 
 
 
 
#Classification 
 
X_cls = df_encoded.drop(columns=['G3', 'pass']) 
y_cls = df_encoded['pass'] 
 
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, 
test_size=0.2, random_state=42) 
 
param_grid_cls = { 
    'n_estimators': [50, 100, 150], 
    'max_depth': [None, 10, 20], 
    'min_samples_split': [2, 5] 
} 
 
rf_cls = RandomForestClassifier(random_state=42) 
grid_cls = GridSearchCV(rf_cls, param_grid=param_grid_cls, cv=5, scoring='accuracy', 
n_jobs=-1) 
grid_cls.fit(X_train_cls, y_train_cls) 
 
best_rf_cls = grid_cls.best_estimator_ 
y_pred_cls = best_rf_cls.predict(X_test_cls) 
 
 
#5. Performance Evaluation 
#Regression Evaluation 
 
print("----- Random Forest Regression -----") 
print("Best Params:", grid_reg.best_params_) 
r2 = r2_score(y_test_reg, y_pred_reg) 
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)) 
print("R2 Score:", r2) 
print("RMSE:", rmse) 
 
#Classification Evaluation 
 
print("\n----- Random Forest Classification -----") 
print("Best Params:", grid_cls.best_params_) 
accuracy = accuracy_score(y_test_cls, y_pred_cls) 
print("Accuracy Score:", accuracy) 
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls)) 
 
#6. Compare the Performance for Different Parameters 
print("----- COMPARISON SUMMARY -----") 
print(f"Random Forest Regression R²: {r2:.3f}, RMSE: {rmse:.3f}") 
print(f"Random Forest Classification Accuracy: {accuracy:.3f}") 
 
if accuracy >= r2: 
    print("\n   BEST ALGORITHM: Random Forest Classification") 
else: 
    print("\n   BEST ALGORITHM: Random Forest Regression") 
 
#7. Data Visualization 
#Bar Plot: School Distribution 
 
school = df['school'].value_counts() 
plt.bar(school.index, school.values, color='purple') 
plt.title('School Distribution') 
plt.xlabel('School') 
plt.ylabel('Count') 
plt.show() 
 
#Absences Distribution (Bar + Pie) 
absences = df['absences'].value_counts() 
plt.bar(absences.index, absences.values, color='purple') 
plt.title('Absences Distribution') 
plt.xlabel('Absences') 
plt.ylabel('Count') 
plt.show() 
 
absences = absences.sort_values(ascending=False) 
threshold = absences.sum() * 0.03 
absences_filtered = absences[absences >= threshold] 
other_sum = absences[absences < threshold].sum() 
if other_sum > 0: 
    absences_filtered["Other"] = other_sum 
 
explode = [0.1 if i < 3 else 0 for i in range(len(absences_filtered))] 
colors = plt.cm.get_cmap("tab20").colors[:len(absences_filtered)] 
 
plt.figure(figsize=(12, 12)) 
wedges, texts, autotexts = plt.pie( 
    absences_filtered, labels=absences_filtered.index, autopct='%0.1f%%', 
    shadow=True, colors=colors, startangle=140, pctdistance=0.6, 
    labeldistance=1.1, explode=explode 
) 
for text in texts: text.set_fontsize(14) 
for autotext in autotexts: autotext.set_fontsize(14) 
plt.title('Absences Distribution', fontsize=18) 
plt.legend(wedges, absences_filtered.index, title="Categories", loc="center left", 
bbox_to_anchor=(1, 0.5), fontsize=12) 
plt.show() 
#Classification Visuals 
 
sns.set(style="whitegrid") 
 
# Count of Pass/Fail 
plt.figure(figsize=(6, 4)) 
sns.countplot(x='pass', data=df_encoded, palette='pastel') 
plt.title('Count of Pass vs Fail') 
plt.xlabel('Pass (1) / Fail (0)') 
plt.ylabel('Number of Students') 
plt.show() 
 
# Histogram: G3 by Pass/Fail 
plt.figure(figsize=(8, 5)) 
sns.histplot(data=df_encoded, x='G3', hue='pass', kde=True, palette='pastel', bins=15) 
plt.title('Distribution of Final Grade (G3) by Pass/Fail') 
plt.xlabel('Final Grade (G3)') 
plt.ylabel('Count') 
plt.show() 
 
# Barplot: Study Time vs Final Grade 
plt.figure(figsize=(8, 5)) 
sns.barplot(x='studytime', y='G3', hue='pass', data=df_encoded, palette='pastel') 
plt.title('Study Time vs Final Grade by Pass/Fail') 
plt.xlabel('Weekly Study Time') 
plt.ylabel('Final Grade (G3)') 
plt.show() 
 
# Actual vs Predicted Grades (Regression) 
plt.figure(figsize=(8, 5)) 
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.6) 
plt.plot([0, 20], [0, 20], 'r--') 
plt.xlabel('Actual G3') 
plt.ylabel('Predicted G3') 
plt.title('Actual vs Predicted Grades (Regression)') 
plt.grid(True) 
plt.show() 
 
# Confusion Matrix 
cm = confusion_matrix(y_test_cls, y_pred_cls) 
plt.figure(figsize=(6, 5)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fail", "Pass"], 
yticklabels=["Fail", "Pass"]) 
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.title("Confusion Matrix - Pass/Fail Classification") 
plt.show() 

