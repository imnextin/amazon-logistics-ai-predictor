import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


''' Pre-processing the data '''

#importing the data 
df = pd.read_excel('data/sample/logistics_clean_sample.csv', engine='openpyxl')


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



'''Exploratory data analysis through data visualization '''

# 1. Distribution of the target variable (Logistics_Delay)
df['Logistics_Delay'].value_counts()

plt.figure(figsize=(12,10))
sns.countplot(data=df, x='Logistics_Delay', palette=['yellowgreen','firebrick'])
plt.title("Distribution of Logistics Delay", fontsize=20, fontweight='bold')
plt.xlabel("Logistics Delay (0 = On-Time, 1 = Delayed)", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# 2. Delayed vs non-delayed shipments across months
#converting the numeric value in the month column to month names 
month_map = {1: "January", 2: "February", 3: "March", 4: "April",5: "May", 6: "June", 7: "July", 8: "August",9: "September", 10: "October", 11: "November", 12: "December"}
df['MonthName'] = df['Month'].map(month_map)

df['DelayStatus'] = df['Logistics_Delay'].map({0: "no delay", 1: "delay"})

month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

plt.figure(figsize=(12,10))
sns.countplot(x='MonthName', hue='DelayStatus', data=df, palette=['salmon','deepskyblue'], order=month_order)
plt.title("Delayed vs Non-Delayed Shipments by Month", fontsize=20,fontweight='bold')
plt.xlabel("Month", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(rotation=45,fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
plt.show()

# 3. Grouped bar chart of Logistics_Delay counts segmented by Traffic_Status
plt.figure(figsize=(12,10))
sns.countplot(x='Traffic_Status' , hue='DelayStatus', data=df, palette=['forestgreen','tomato'], hue_order=["no delay", "delay"])
plt.title("Logistics Delay by Traffic Status", fontsize=20,fontweight='bold')
plt.xlabel("Traffic Status", fontsize=18)
plt.ylabel("Number of Shipments", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(title="Delay Status",fontsize=18)
plt.show()

# 4. Box plot comparing Waiting_Time across the two Logistics_Delay classes
plt.figure(figsize=(12,10))
sns.boxplot(x='DelayStatus', y='Waiting_Time', data=df, palette=['seagreen','lightpink'], order=["no delay", "delay"])
plt.title("Waiting Time by Delay Status", fontsize=20,fontweight='bold')
plt.xlabel("Delay Status", fontsize=18)
plt.ylabel("Waiting Time", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=18)
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

''' Feature Selection ''' 

#Initialize 
logmodel= LogisticRegression(solver='liblinear')

#Fitting the model 
logmodel.fit(x_train, y_train)

#Evaluating the model 
y_predict=logmodel.predict(x_test)
score=f1_score(y_test, y_predict)
print ("The F1 score is", score) #1

#Initializing feature selection 
sfs=SFS(logmodel, k_features=(1, 14), forward=True, scoring='f1', cv=5) 
sfs.fit(x_train,y_train)

#Features selected 
sfs.k_feature_names_

#Transforming the data with selected features 
x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

#Fitting the model with new features 
logmodel.fit(x_train_sfs,y_train)

#Evaluating the model 
y_pred=logmodel.predict(x_test_sfs)
score1 = f1_score(y_test,y_pred) #1
print ("The F1 score is", score1)  

cm3=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm3)

corr= df.corr() #correlation table 

#The reason behind the perfect f1 score above is the inclusion of highly correlated variable Traffic_Status_Heavy=+0.62
#Hence, to improve the predictive model, removal of this variable is important 

##removing Traffic_Status and running feature selection to improve the model 

x = df.drop(columns=['Logistics_Delay', 'Traffic_Status'], errors='ignore')
y = df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1) 

#Initialize 
logmodel= LogisticRegression(solver='liblinear')

#Fitting the model 
logmodel.fit(x_train, y_train)

#Evaluating the model 
y_predict=logmodel.predict(x_test)
score=f1_score(y_test, y_predict)
print ("The F1 score is", score) #0.80

#Initializing feature selection 
sfs=SFS(logmodel, k_features=(1, 13), forward=True, scoring='f1', cv=5) 
sfs.fit(x_train,y_train)

#Features selected 
sfs.k_feature_names_

#Transforming the data with selected features 
x_train_sfs=sfs.transform(x_train)
x_test_sfs=sfs.transform(x_test)

#Fitting the model with new features 
logmodel.fit(x_train_sfs,y_train)

#Evaluating the model 
y_pred=logmodel.predict(x_test_sfs)
score1 = f1_score(y_test,y_pred) 
print ("The F1 score is", score1) #0.78

cm2=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm2)




''' Logistic Regression '''

#Building model - Attemp 1 
#Building a logistic regression model with the variables chosen from feature selection 

x1=df[['Latitude','Temperature','Shipment_Status','Traffic_Status','Logistics_Delay_Reason']]
y=df['Logistics_Delay']

x1_train,x1_test,y_train,y_test=train_test_split(x1,y,test_size=0.3,random_state=1)

logmodel1= LogisticRegression(solver='liblinear') #initialize 

logmodel1.fit(x1_train,y_train) #train
logmodel1.intercept_
logmodel1.coef_

probabilities1=logmodel1.predict_proba(x1_test) #predict 
prediction1=logmodel1.predict(x1_test)

#Evaluating the model based on recall, precision, f1 scores and confusion matrix 

f1_score(y_test, prediction1) #0.80

recall_score(y_test, prediction1) #0.72

precision_score(y_test, prediction1) #0.91

con_max=pd.DataFrame(confusion_matrix(y_test,prediction1,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(con_max) 

#Building model - Attempt 2 

x = df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'], errors='ignore')
y = df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

logmodel= LogisticRegression(solver='liblinear') #initialize 

logmodel.fit(x_train,y_train) #train
logmodel.intercept_
logmodel.coef_

probabilities=logmodel.predict_proba(x_test) #predict 
prediction=logmodel.predict(x_test)

#Evaluating the model based on recall, precision, f1 scores and confusion matrix 

f1_score(y_test, prediction) #0.80

recall_score(y_test, prediction) #0.71

precision_score(y_test, prediction) #0.91

cm_lr=pd.DataFrame(confusion_matrix(y_test,prediction,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_lr) 

#changing the threshold from 0.5 to 0.3
probabilities=logmodel.predict_proba(x_test)

y_probs=probabilities[:,1]

y_pred3=np.where(y_probs>0.3,1,0)

cm_lr1=pd.DataFrame(confusion_matrix(y_test,y_pred3,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_lr1) 

recall_score(y_test, y_pred3) #0.95 

f1_score(y_test, y_pred3) #0.87

precision_score(y_test, y_pred3) #0.8
 
#This model gives the least number of False Negative, so it makes more sense to go with this model 




'''Decision tree'''

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

dt=DecisionTreeClassifier(random_state=1) #initialize

dt.fit(x_train,y_train) #train

dt.tree_.max_depth

#F1 score for training data 
tree_pred_train=dt.predict(x_train)
f1_score(y_train,tree_pred_train) #f1 = 1

tree_pred_test=dt.predict(x_test)
f1_score(y_test,tree_pred_test) #f1 = 0.80 

cm_dt1=pd.DataFrame(confusion_matrix(y_test,tree_pred_test,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_dt1)

#Tuning the hyper parameter 
parameter_grid={'max_depth':range(1,14),'min_samples_split':range(2,20)}

grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring='f1',cv=5) #initialize 
grid.fit(x_train,y_train)

grid.best_params_ #best hyperparameters 
#{'max_depth': 14, 'min_samples_split': 2}

#building the model with tuned hyper parameters 
dt=DecisionTreeClassifier(max_depth=12, min_samples_split=2, random_state=1)
dt.fit(x_train,y_train)

#Training F1 score after tuning the model 
train_tuned=dt.predict(x_train)
f1_score(y_train,train_tuned) #f1 = 0.99 ; dropped from 1 to 0.99 

#Testing F1 score after tuning the model 
test_tuned=dt.predict(x_test)
f1_score(y_test,test_tuned) #f1 = 0.80 ; remained same 

#the decision tree is at optimal functioning hence no noticiable improvement in the f1 score in the train and test data 

cm_dt=pd.DataFrame(confusion_matrix(y_test,test_tuned,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_dt) 

recall_score(y_test,test_tuned) #0.74

precision_score(y_test,test_tuned) #0.86



'''Random Forest'''

from sklearn.ensemble import RandomForestClassifier

x = df.drop(columns=['Logistics_Delay', 'Traffic_Status_Heavy'], errors='ignore')
y = df['Logistics_Delay']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

rf=RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(x_train,y_train)

#Training F1 score 
train_rf=rf.predict(x_train)
print("Training F1 Score:", f1_score(y_train, train_rf)) #1.0

#Testing F1 score 
test_rf=rf.predict(x_test)
print("Training F1 Score:", f1_score(y_test, test_rf)) #0.82

cm_rf=pd.DataFrame(confusion_matrix(y_test,test_rf,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_rf) 

recall_score(y_test,test_rf) #0.75

precision_score(y_test,test_rf) #0.89
#Tuning hyperparameters for random forest 

param_grid = {'n_estimators': [50, 100, 200],'max_depth': [None, 5, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['sqrt', 'log2']}

grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='f1',n_jobs=-1,verbose=1)

grid_search.fit(x_train,y_train)

#Best parameters and model 
print("Best Parameters:", grid_search.best_params_)
best_rf=grid_search.best_estimator_

#Evaluate on training set
train_pred_rfs=best_rf.predict(x_train)
print("Training F1 Score:", f1_score(y_train, train_pred_rfs)) #0.94

#Evaluate on testing set
test_pred_rfs=best_rf.predict(x_test)
print("Training F1 Score:", f1_score(y_test, test_pred_rfs)) #0.82

cm_rf1=pd.DataFrame(confusion_matrix(y_test,test_pred_rfs,labels=[0,1]),index=["Actual:0", "Actual:1"],columns=["Pred:0",'Pred:1'])
print(cm_rf1) 

recall_score(y_test,test_pred_rfs) #0.78

precision_score(y_test,test_pred_rfs) #0.87


'''After comparing all the three models, we choose to go with the logtistic regression model because it 
gives the least false negative cases'''
 


# Legacy single-row `predict` calls removed (shape/feature mismatch).
# Use the model evaluation block below for proper multi-class evaluation.

''' Multi-class classification: On-Time (0), At Risk (1), Delayed (2) '''

# Work on a copy so earlier binary code remains for reference
df_mc = df.copy()

# --- Feature engineering / proxies
# Orders per timestamp (order volume proxy)
if 'Timestamp' in df_mc.columns:
	df_mc['OrdersPerTimestamp'] = df_mc.groupby('Timestamp')['Timestamp'].transform('count')
else:
	df_mc['OrdersPerTimestamp'] = 1

# Distance proxy: euclidean distance from mean location
if {'Latitude','Longitude'}.issubset(df_mc.columns):
	df_mc['DistanceProxy'] = np.sqrt((df_mc['Latitude'] - df_mc['Latitude'].mean())**2 + (df_mc['Longitude'] - df_mc['Longitude'].mean())**2)
else:
	df_mc['DistanceProxy'] = 0

# Historical delivery performance: avg delay rate by shipment status
if 'Shipment_Status' in df_mc.columns and 'Logistics_Delay' in df_mc.columns:
	hist = df_mc.groupby('Shipment_Status')['Logistics_Delay'].transform('mean')
	df_mc['HistDelayByStatus'] = hist
else:
	df_mc['HistDelayByStatus'] = 0

# Use Waiting_Time as a proxy for severity; create multiclass label using simple thresholds
# If Logistics_Delay==0 -> On-Time (0); if Logistics_Delay==1 and Waiting_Time <= 0.5 -> At Risk (1), else Delayed (2)
threshold = 0.5
def make_status(row):
	ld = int(row.get('Logistics_Delay', 0))
	wt = row.get('Waiting_Time', 0)
	if ld == 0:
		return 0
	else:
		return 1 if wt <= threshold else 2

df_mc['Delivery_Status'] = df_mc.apply(make_status, axis=1)

print('\nMulti-class distribution:')
print(df_mc['Delivery_Status'].value_counts())

# Prepare features and target
X = df_mc.drop(columns=['Logistics_Delay','Delivery_Status'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)
y = df_mc['Delivery_Status']

from sklearn.model_selection import train_test_split
# stratify to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Handle class imbalance: try SMOTE, fall back to no resampling and class_weight
use_resample = False
try:
	from imblearn.over_sampling import SMOTE
	sm = SMOTE(random_state=1)
	X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
	use_resample = True
	print('\nApplied SMOTE; training class counts:', np.bincount(y_train_res))
except Exception as e:
	print('\nSMOTE unavailable or failed (', e, '). Using original training set and class_weight balanced.')
	X_train_res, y_train_res = X_train, y_train

# Models to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = {
	'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced'),
	'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=1),
	'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=1)
}
try:
	from xgboost import XGBClassifier
	models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=1)
except Exception as e:
	print('XGBoost not available (', e, '); skipping XGBoost.')

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

for name, model in models.items():
	print('\nTraining', name)
	model.fit(X_train_res, y_train_res)
	preds = model.predict(X_test)
	print('Model:', name)
	print('Confusion Matrix:\n', confusion_matrix(y_test, preds, labels=[0,1,2]))
	print('\nClassification Report:\n', classification_report(y_test, preds, digits=4))
	p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='macro')
	print('Macro Precision: %.4f, Macro Recall: %.4f, Macro F1: %.4f' % (p, r, f))

print('\nMulti-class modeling complete.\n')






