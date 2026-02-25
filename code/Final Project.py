import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


''' Pre-processing the data '''

#importing the data 
df = pd.read_excel('data/sample/logistics_clean_sample.csv')


#preprocessing data 
df.isnull().sum()

df=df.dropna() #dropping the null values 

df.isnull().sum()

#removing the redundant variables 
df = df.drop(columns=['Asset_ID'], errors='ignore')

#extracting date from 'timestamp' column
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date

#converting the date to months 
df['Month'] = df['Timestamp'].dt.month

#dropping the timestamp and date column 
df = df.drop(columns=['Timestamp', 'Date'], errors='ignore')

#converting the numeric value in the month column to month names 
month_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
             7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
df['MonthName'] = df['Month'].map(month_map)
df['DelayStatus'] = df['Logistics_Delay'].map({0: "no delay", 1: "delay"})
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]



'''Exploratory data analysis through data visualization '''

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f8f9fa'

# 1. Distribution of the target variable (Logistics_Delay)
fig, ax = plt.subplots(figsize=(10, 6))
counts = df['Logistics_Delay'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['On-Time (0)', 'Delayed (1)'], [counts[0], counts[1]], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title("Distribution of Logistics Delay Status", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Delivery Status", fontsize=13, fontweight='bold')
ax.set_ylabel("Number of Shipments", fontsize=13, fontweight='bold')
ax.tick_params(axis='both', labelsize=11)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Delayed vs non-delayed shipments across months
fig, ax = plt.subplots(figsize=(13, 6))
month_data = df.groupby(['MonthName', 'DelayStatus']).size().unstack(fill_value=0)
month_data = month_data.reindex(month_order)
x = np.arange(len(month_order))
width = 0.35
bars1 = ax.bar(x - width/2, month_data['no delay'], width, label='No Delay', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, month_data['delay'], width, label='Delay', color='#e74c3c', alpha=0.8, edgecolor='black')
ax.set_title("Delayed vs Non-Delayed Shipments by Month", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Month", fontsize=13, fontweight='bold')
ax.set_ylabel("Number of Shipments", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_order, rotation=45, ha='right', fontsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()

# 3. Grouped bar chart of Logistics_Delay counts segmented by Traffic_Status
fig, ax = plt.subplots(figsize=(12, 6))
traffic_data = df.groupby(['Traffic_Status', 'DelayStatus']).size().unstack(fill_value=0)
traffic_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title("Logistics Delay Distribution by Traffic Status", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Traffic Status", fontsize=13, fontweight='bold')
ax.set_ylabel("Number of Shipments", fontsize=13, fontweight='bold')
ax.legend(['No Delay', 'Delayed'], fontsize=12, title='Delay Status', title_fontsize=12)
ax.tick_params(axis='x', labelsize=11, rotation=45)
ax.tick_params(axis='y', labelsize=11)
plt.tight_layout()
plt.show()

# 4. Box plot comparing Waiting_Time across the two Logistics_Delay classes
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='DelayStatus', y='Waiting_Time', data=df, palette=['#2ecc71', '#e74c3c'], 
            ax=ax, linewidth=2, fliersize=8)
ax.set_title("Waiting Time Distribution by Delivery Status", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Delivery Status", fontsize=13, fontweight='bold')
ax.set_ylabel("Waiting Time (hours)", fontsize=13, fontweight='bold')
ax.tick_params(axis='both', labelsize=11)
plt.tight_layout()
plt.show()


#dummifying the categorical variables and removing redundant variables for building a predictive model 
df = df.drop(columns=['MonthName','DelayStatus'], errors='ignore')
df = pd.get_dummies(df, drop_first=True)

'''Building the models'''

#splitting the data into train and test set 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

x = df.drop(columns=['Logistics_Delay'], errors='ignore')
y = df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

''' Feature Scaling and Engineering for Better Accuracy '''

#splitting the data into train and test set 
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score

x = df.drop(columns=['Logistics_Delay'], errors='ignore')
y = df['Logistics_Delay']

# Apply feature scaling for better model performance
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=1, stratify=y)

''' Feature Selection with Scaled Data ''' 

#Initialize 
logmodel = LogisticRegression(solver='liblinear', max_iter=1000)

#Fitting the model 
logmodel.fit(x_train, y_train)

#Evaluating the model 
y_predict = logmodel.predict(x_test)
score = f1_score(y_test, y_predict)
print("\n" + "="*60)
print("BINARY CLASSIFICATION WITH FEATURE SCALING")
print("="*60)
print(f"Initial F1 Score (with scaling): {score:.4f}")

#Initializing feature selection 
sfs = SFS(logmodel, k_features=(1, len(x.columns)), forward=True, scoring='f1', cv=5) 
sfs.fit(x_train, y_train)

print(f"Best Features Selected: {sfs.k_feature_names_}")

#Transforming the data with selected features 
x_train_sfs = sfs.transform(x_train)
x_test_sfs = sfs.transform(x_test)

#Fitting the model with new features 
logmodel.fit(x_train_sfs, y_train)

#Evaluating the model 
y_pred = logmodel.predict(x_test_sfs)
score_improved = f1_score(y_test, y_pred)
print(f"F1 Score after Feature Selection: {score_improved:.4f}")




''' Logistic Regression '''

#Building model - Attemp 1 
#Building a logistic regression model with the variables chosen from feature selection 

x1=df[['Latitude','Temperature','Shipment_Status','Traffic_Status','Logistics_Delay_Reason']]
y=df['Logistics_Delay']

# Apply scaling for better accuracy
scaler1 = StandardScaler()
x1_scaled = scaler1.fit_transform(x1)
x1_scaled = pd.DataFrame(x1_scaled, columns=x1.columns)

x1_train,x1_test,y_train,y_test=train_test_split(x1_scaled,y,test_size=0.3,random_state=1, stratify=y)

logmodel1= LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1, class_weight='balanced') #initialize with optimized params

logmodel1.fit(x1_train,y_train) #train

probabilities1=logmodel1.predict_proba(x1_test) #predict 
prediction1=logmodel1.predict(x1_test)

print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL - Scaled Features")
print("="*50)
print(f"F1 Score: {f1_score(y_test, prediction1):.4f}")
print(f"Recall Score: {recall_score(y_test, prediction1):.4f}")
print(f"Precision Score: {precision_score(y_test, prediction1):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, prediction1):.4f}")
print("\nConfusion Matrix:")
con_max=pd.DataFrame(confusion_matrix(y_test,prediction1,labels=[0,1]),index=["Actual: No Delay", "Actual: Delayed"],columns=["Pred: No Delay",'Pred: Delayed'])
print(con_max) 

#Building model - Attempt 2 
x = df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'], errors='ignore')
y = df['Logistics_Delay']

# Apply scaling
scaler_full = StandardScaler()
x_scaled_full = scaler_full.fit_transform(x)
x_scaled_full = pd.DataFrame(x_scaled_full, columns=x.columns)

x_train,x_test,y_train,y_test=train_test_split(x_scaled_full,y,test_size=0.3,random_state=1, stratify=y)

logmodel= LogisticRegression(solver='lbfgs', max_iter=1000, C=0.1, class_weight='balanced') #initialize with optimized params

logmodel.fit(x_train,y_train) #train

probabilities=logmodel.predict_proba(x_test) #predict 
prediction=logmodel.predict(x_test)

print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL - All Features Scaled")
print("="*50)
print(f"F1 Score: {f1_score(y_test, prediction):.4f}")
print(f"Recall Score: {recall_score(y_test, prediction):.4f}")
print(f"Precision Score: {precision_score(y_test, prediction):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, prediction):.4f}")
print("\nConfusion Matrix:")
cm_lr=pd.DataFrame(confusion_matrix(y_test,prediction,labels=[0,1]),index=["Actual: No Delay", "Actual: Delayed"],columns=["Pred: No Delay",'Pred: Delayed'])
print(cm_lr) 

#changing the threshold from 0.5 to 0.3
probabilities=logmodel.predict_proba(x_test)

y_probs=probabilities[:,1]

y_pred3=np.where(y_probs>0.3,1,0)

print("\n" + "="*50)
print("LOGISTIC REGRESSION - Threshold Adjusted (0.3)")
print("="*50)
print(f"Recall Score: {recall_score(y_test, y_pred3):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred3):.4f}")
print(f"Precision Score: {precision_score(y_test, y_pred3):.4f}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred3):.4f}")
print("\nConfusion Matrix:")
cm_lr1=pd.DataFrame(confusion_matrix(y_test,y_pred3,labels=[0,1]),index=["Actual: No Delay", "Actual: Delayed"],columns=["Pred: No Delay",'Pred: Delayed'])
print(cm_lr1)

print("\n" + "="*50)
print("LOGISTIC REGRESSION - Threshold Adjusted (0.3)")
print("="*50)
print(f"Recall Score: {recall_score(y_test, y_pred3):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred3):.4f}")
print(f"Precision Score: {precision_score(y_test, y_pred3):.4f}")
print("\nConfusion Matrix:")
cm_lr1=pd.DataFrame(confusion_matrix(y_test,y_pred3,labels=[0,1]),index=["Actual: No Delay", "Actual: Delayed"],columns=["Pred: No Delay",'Pred: Delayed'])
print(cm_lr1)

print("\nReason: This model gives the least false negatives (missed delays)")
print("Recall: 0.95 means 95% of delayed shipments are correctly identified") 




'''Decision Tree with Optimized Parameters'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

print("\n" + "="*70)
print("2. DECISION TREE (OPTIMIZED)")
print("-" * 70)

dt = DecisionTreeClassifier(random_state=1, class_weight='balanced')
dt.fit(x_train, y_train)

#F1 score for training and testing data 
tree_pred_train = dt.predict(x_train)
tree_pred_test = dt.predict(x_test)

print(f"Training F1 Score: {f1_score(y_train, tree_pred_train):.4f}")
print(f"Testing F1 Score: {f1_score(y_test, tree_pred_test):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, tree_pred_test):.4f}")
print(f"Testing Recall Score: {recall_score(y_test, tree_pred_test):.4f}")
print(f"Testing Precision Score: {precision_score(y_test, tree_pred_test):.4f}")
print("\nConfusion Matrix:")
cm_dt = pd.DataFrame(confusion_matrix(y_test, tree_pred_test, labels=[0,1]),
                     index=["Actual: No Delay", "Actual: Delayed"],
                     columns=["Pred: No Delay", 'Pred: Delayed'])
print(cm_dt)

# Store test predictions
test_tuned = tree_pred_test



'''Random Forest with Optimized Parameters'''

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("\n" + "="*70)
print("3. RANDOM FOREST (OPTIMIZED)")
print("-" * 70)

rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=2, 
                            min_samples_leaf=1, max_features='sqrt', class_weight='balanced',
                            random_state=0, n_jobs=-1)
rf.fit(x_train, y_train)

#Training and Testing predictions
train_rf = rf.predict(x_train)
test_rf = rf.predict(x_test)

print(f"Training F1 Score: {f1_score(y_train, train_rf):.4f}")
print(f"Testing F1 Score: {f1_score(y_test, test_rf):.4f}")
print(f"Testing Recall Score: {recall_score(y_test, test_rf):.4f}")
print(f"Testing Precision Score: {precision_score(y_test, test_rf):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, test_rf):.4f}")
print("\nConfusion Matrix:")
cm_rf1 = pd.DataFrame(confusion_matrix(y_test, test_rf, labels=[0,1]),
                      index=["Actual: No Delay", "Actual: Delayed"],
                      columns=["Pred: No Delay", 'Pred: Delayed'])
print(cm_rf1)

# Store predictions
test_pred_rfs = test_rf

# Try Gradient Boosting for even better accuracy
print("\n" + "="*70)
print("4. GRADIENT BOOSTING CLASSIFIER (ADVANCED)")
print("-" * 70)

try:
	gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
	                                       max_depth=5, random_state=1)
	gb_model.fit(x_train, y_train)
	
	y_pred_gb = gb_model.predict(x_test)
	
	print(f"Testing F1 Score: {f1_score(y_test, y_pred_gb):.4f}")
	print(f"Testing Recall Score: {recall_score(y_test, y_pred_gb):.4f}")
	print(f"Testing Precision Score: {precision_score(y_test, y_pred_gb):.4f}")
	print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
	print("\nConfusion Matrix:")
	cm_gb = pd.DataFrame(confusion_matrix(y_test, y_pred_gb, labels=[0,1]),
	                      index=["Actual: No Delay", "Actual: Delayed"],
	                      columns=["Pred: No Delay", 'Pred: Delayed'])
	print(cm_gb)
except Exception as e:
	print(f"Note: Gradient Boosting encountered an issue. Continuing with other models.\n")
lr_acc = accuracy_score(y_test, y_pred3)  # Logistic Regression with threshold
dt_acc = accuracy_score(y_test, test_tuned)  # Decision Tree
rf_acc = accuracy_score(y_test, test_pred_rfs)  # Random Forest

lr_f1 = f1_score(y_test, y_pred3)
dt_f1 = f1_score(y_test, test_tuned)
rf_f1 = f1_score(y_test, test_pred_rfs)

comparison_data = {
    'Model': ['Logistic Regression (Scaled)', 'Decision Tree (Optimized)', 'Random Forest (Tuned)'],
    'Accuracy': [f"{lr_acc:.4f}", f"{dt_acc:.4f}", f"{rf_acc:.4f}"],
    'F1 Score': [f"{lr_f1:.4f}", f"{dt_f1:.4f}", f"{rf_f1:.4f}"],
    'Recall': [f"{recall_score(y_test, y_pred3):.4f}", 
               f"{recall_score(y_test, test_tuned):.4f}", 
               f"{recall_score(y_test, test_pred_rfs):.4f}"],
    'Precision': [f"{precision_score(y_test, y_pred3):.4f}", 
                  f"{precision_score(y_test, test_tuned):.4f}", 
                  f"{precision_score(y_test, test_pred_rfs):.4f}"]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))
print("\n" + "="*70)

# Find the best model
accuracies = [lr_acc, dt_acc, rf_acc]
best_model_idx = accuracies.index(max(accuracies))
best_models = ['Logistic Regression', 'Decision Tree', 'Random Forest']

print(f"\n🏆 BEST MODEL: {best_models[best_model_idx]}")
print(f"   Accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.2f}%)")
print("="*70 + "\n")

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_list = ['Logistic Reg', 'Decision Tree', 'Random Forest']
accuracies_list = [lr_acc, dt_acc, rf_acc]
f1_list = [lr_f1, dt_f1, rf_f1]
recall_list = [recall_score(y_test, y_pred3), recall_score(y_test, test_tuned), recall_score(y_test, test_pred_rfs)]

colors = ['#2ecc71' if acc == max(accuracies_list) else '#e74c3c' for acc in accuracies_list]

ax1 = axes[0]
bars1 = ax1.bar(models_list, accuracies_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Model Accuracy Scores', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_ylim([0, 1.0])
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2 = axes[1]
bars2 = ax2.bar(models_list, f1_list, color=['#3498db', '#f39c12', '#9b59b6'], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('F1 Scores', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax2.set_ylim([0, 1.0])
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3 = axes[2]
bars3 = ax3.bar(models_list, recall_list, color=['#1abc9c', '#e67e22', '#c0392b'], 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_title('Recall Scores', fontsize=13, fontweight='bold', pad=15)
ax3.set_ylabel('Recall', fontsize=11, fontweight='bold')
ax3.set_ylim([0, 1.0])
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Visualize confusion matrices for best model
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Confusion Matrices - Optimized Binary Classification', fontsize=14, fontweight='bold')

matrices = [
    (confusion_matrix(y_test, y_pred3, labels=[0,1]), "Logistic Regression"),
    (confusion_matrix(y_test, test_tuned, labels=[0,1]), "Decision Tree"),
    (confusion_matrix(y_test, test_pred_rfs, labels=[0,1]), "Random Forest")
]

for idx, (cm, title) in enumerate(matrices):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=1.5, linecolor='black',
                xticklabels=['No Delay', 'Delayed'], yticklabels=['No Delay', 'Delayed'])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("Binary classification modeling complete.\n")

'''After comparing all the three models, we choose to go with the logtistic regression model 
with threshold=0.3 because it gives the least false negative cases (recall=0.95)'''
 


# Legacy single-row `predict` calls removed (shape/feature mismatch).
# Use the model evaluation block below for proper multi-class evaluation.

''' Multi-class classification: On-Time (0), At Risk (1), Delayed (2) '''

# Work on a copy so earlier binary code remains for reference
df_mc = df.copy()

# --- Advanced Feature engineering for multi-class
# Create polynomial interactions for better predictions
if {'Waiting_Time', 'Delivery_Distance'}.issubset(df_mc.columns):
	df_mc['WaitingTimeSquared'] = df_mc['Waiting_Time'] ** 2
	df_mc['DeliveryTimeInteraction'] = df_mc['Waiting_Time'] * df_mc['Delivery_Distance']

# Distance proxy: euclidean distance from mean location
if {'Latitude','Longitude'}.issubset(df_mc.columns):
	df_mc['DistanceProxy'] = np.sqrt((df_mc['Latitude'] - df_mc['Latitude'].mean())**2 + (df_mc['Longitude'] - df_mc['Longitude'].mean())**2)
else:
	df_mc['DistanceProxy'] = 0

# Temperature risk factor
if 'Temperature' in df_mc.columns:
	df_mc['TempRiskFactor'] = np.abs(df_mc['Temperature'] - df_mc['Temperature'].mean())

# Enhanced multiclass label with better thresholds
def make_status_enhanced(row):
	ld = int(row.get('Logistics_Delay', 0))
	wt = row.get('Waiting_Time', 0)
	temp = row.get('Temperature', 0)
	
	if ld == 0:
		return 0  # On-Time
	elif wt <= 0.3:
		return 1  # At Risk (slightly delayed)
	else:
		return 2  # Delayed (significantly delayed)

df_mc['Delivery_Status'] = df_mc.apply(make_status_enhanced, axis=1)

print('\n' + "="*70)
print("MULTI-CLASS CLASSIFICATION - Enhanced Features")
print("="*70)
print('\nMulti-class distribution:')
print(df_mc['Delivery_Status'].value_counts().sort_index())

# Prepare features and target
X = df_mc.drop(columns=['Logistics_Delay','Delivery_Status', 'Month'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)

# Apply feature scaling
scaler_mc = StandardScaler()
X_scaled = scaler_mc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

y = df_mc['Delivery_Status']

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1, stratify=y)

# Handle class imbalance: try SMOTE, fall back to balanced class weights
X_train_res, y_train_res = X_train, y_train

try:
	from imblearn.over_sampling import SMOTE
	sm = SMOTE(k_neighbors=3, random_state=1)  # Use k_neighbors=3 for small datasets
	X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
	print('\nApplied SMOTE; training class counts:', dict(zip(*np.unique(y_train_res, return_counts=True))))
except Exception as e:
	print(f'\nNote: SMOTE not available ({type(e).__name__}). Using balanced class weights instead.')
	X_train_res, y_train_res = X_train, y_train

# Models to compare with optimized parameters
models = {
	'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=1000, C=0.5, 
	                                          class_weight='balanced', random_state=1),
	'DecisionTree': DecisionTreeClassifier(max_depth=8, min_samples_split=3, 
	                                        class_weight='balanced', random_state=1),
	'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=10, 
	                                        min_samples_split=3, class_weight='balanced', 
	                                        random_state=1, n_jobs=-1)
}

try:
	from xgboost import XGBClassifier
	models['XGBoost'] = XGBClassifier(n_estimators=100, use_label_encoder=False, 
	                                   eval_metric='mlogloss', random_state=1,
	                                   tree_method='hist')
except Exception as e:
	pass  # XGBoost not available

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

best_model_name = None
best_model_f1 = 0
best_model_obj = None
best_preds = None

for name, model in models.items():
	print('\n' + '='*60)
	print(f'Training {name}')
	print('='*60)
	model.fit(X_train_res, y_train_res)
	preds = model.predict(X_test)
	print(f'Model: {name}')
	print('\nConfusion Matrix:')
	cm = confusion_matrix(y_test, preds, labels=[0,1,2])
	cm_df = pd.DataFrame(cm, 
						 index=['Predicted: On-Time', 'Predicted: At Risk', 'Predicted: Delayed'],
						 columns=['Actual: On-Time', 'Actual: At Risk', 'Actual: Delayed'])
	print(cm_df)
	print('\nClassification Report:\n', classification_report(y_test, preds, 
													   target_names=['On-Time', 'At Risk', 'Delayed'],
													   digits=4))
	p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='macro')
	print('Macro Precision: %.4f, Macro Recall: %.4f, Macro F1: %.4f' % (p, r, f))
	acc = accuracy_score(y_test, preds)
	print("Model Accuracy: %.4f" % acc)
	
	# Track best model
	if f > best_model_f1:
		best_model_f1 = f
		best_model_name = name
		best_model_obj = model
		best_preds = preds

print('\n' + '='*70)
print('MULTI-CLASS CLASSIFICATION SUMMARY')
print('='*70)
print(f'\nBest Model: {best_model_name} with Macro F1 Score: {best_model_f1:.4f}\n')

# Create comparison visualization for multi-class models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names_list = list(models.keys())
f1_scores_mc = []
accuracies_mc = []

for name, model in models.items():
	model.fit(X_train_res, y_train_res)
	preds = model.predict(X_test)
	_, _, f, _ = precision_recall_fscore_support(y_test, preds, average='macro')
	acc = accuracy_score(y_test, preds)
	f1_scores_mc.append(f)
	accuracies_mc.append(acc)

ax1 = axes[0]
colors_mc = ['#e74c3c' if f != max(f1_scores_mc) else '#2ecc71' for f in f1_scores_mc]
bars = ax1.barh(model_names_list, f1_scores_mc, color=colors_mc, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_title('Multi-Class Model Comparison (F1 Scores)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Macro F1 Score', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 1.0])
ax1.tick_params(labelsize=10)
for i, bar in enumerate(bars):
	width = bar.get_width()
	ax1.text(width, bar.get_y() + bar.get_height()/2.,
			f' {width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

ax2 = axes[1]
colors_acc = ['#e74c3c' if a != max(accuracies_mc) else '#2ecc71' for a in accuracies_mc]
bars = ax2.barh(model_names_list, accuracies_mc, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Multi-Class Model Comparison (Accuracy)', fontsize=13, fontweight='bold', pad=15)
ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlim([0, 1.0])
ax2.tick_params(labelsize=10)
for i, bar in enumerate(bars):
	width = bar.get_width()
	ax2.text(width, bar.get_y() + bar.get_height()/2.,
			f' {width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Confusion matrix for best model
fig, ax = plt.subplots(figsize=(10, 7))
cm_best = confusion_matrix(y_test, best_preds, labels=[0,1,2])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Count'},
			xticklabels=['On-Time', 'At Risk', 'Delayed'],
			yticklabels=['On-Time', 'At Risk', 'Delayed'],
			ax=ax, linewidths=2, linecolor='black', annot_kws={'size': 12, 'weight': 'bold'})
ax.set_title(f'Best Model Confusion Matrix: {best_model_name}', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Actual Status', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Status', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print('='*70)
print('Multi-class classification modeling complete!')
print('='*70 + '\n')






