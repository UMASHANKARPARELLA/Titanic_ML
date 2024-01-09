#!/usr/bin/env python
# coding: utf-8

# # Project Introduction

# - To develop an accurate machine learning model that predicts survival using the Titanic dataset.
# - To analyze how various factors such as age, gender, ticket class, and the number of accompanying family members impacted survival rates.
# - To provide insights that can aid in evaluating the efficiency of life-saving and safety measures in disaster situations.

# # EDA & Preprocessing

# In[1]:


# Import necessary modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
# %config InlineBackend.figure_format = 'retina' ## This is preferable for retina display. 

import warnings ## importing warnings library. 
warnings.filterwarnings('ignore') ## Ignore warning


# In[2]:


## Importing the datasets
train = pd.read_csv("Titanic train.csv")
test = pd.read_csv("Titanic test.csv")


# Train dataset overview

# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


train.describe()


# In[6]:


train.isnull().sum()


# Drop 'Cabin' column - too many null values

# In[7]:


# Drop 'Cabin' column - too many null values
train.drop('Cabin', axis=1, inplace=True)


# In[8]:


# Age distribution
import plotly.express as px

fig = px.box(train, y='Age', title='Boxplot of Age in Train Dataset')

fig.show()


# In[9]:


# Count number of outliers
Q1 = train['Age'].quantile(0.25)
Q3 = train['Age'].quantile(0.75)
IQR = Q3 - Q1

outliers = train[(train['Age'] < (Q1 - 1.5 * IQR)) | (train['Age'] > (Q3 + 1.5 * IQR))]
outlier_count = outliers.shape[0]

print(f"Number of outliers in 'Age': {outlier_count}")


# In[10]:


# Replace null values in 'Age' column with median value: 28.
train['Age'].fillna(28, inplace=True)


# In[11]:


# Delete rows with null values in 'Embarked' column
train.dropna(subset=['Embarked'], inplace=True)


# In[12]:


train.isnull().sum()


# In[13]:


import matplotlib.pyplot as plt

for column in train.columns:
    if train[column].dtype in ['int64', 'float64']:
        train[column].hist(bins=20)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()
    else:
        train[column].value_counts().plot(kind='bar')
        plt.title(f'{column} Value Counts')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# In[14]:


import seaborn as sns

numeric_train = train.select_dtypes(include=[np.number])

correlation = numeric_train.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation between Features and Survival')
plt.show()


# Relationship b/w target value and categorical features

# In[15]:


categorical_features = ['Pclass', 'Sex', 'Embarked']

for feature in categorical_features:
    sns.barplot(x=feature, y='Survived', data=train)
    plt.title(f'Survival Rate by {feature}')
    plt.show()


# Relationship b/w target and age

# In[16]:


sns.kdeplot(data=train, x='Age', hue='Survived', shade=True)
plt.title('Survival Distribution by Age')
plt.show()


# # Feature Engineering

# Age with 4 groups

# In[17]:


bins_4 = [0, 13, 18, 65, 81]
labels_4 = [1, 2, 3, 4]

train['Age'] = pd.cut(train['Age'], bins=bins_4, labels=labels_4, right=False)
train['Age'] = train['Age'].astype(int)


# In[18]:


train.head()


# In[19]:


import re

def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(extract_title)

title_counts = train['Title'].value_counts()

print(title_counts)


# In[20]:


def map_title_to_group(title):
    if title in ['Mr', 'Miss', 'Mrs', 'Ms']:
        return 1
    elif title in ['Don', 'Sir', 'Countess', 'Lady', 'Jonkheer']:
        return 2
    elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
        return 3
    elif title in ['Master', 'Mlle', 'Mme']:
        return 4
    else:
        return 5

train['Group'] = train['Title'].apply(map_title_to_group)


# In[21]:


train.head()


# Fare into 7 groups

# In[22]:


fig = px.box(train, y='Fare', title='Boxplot of Fare in Train Dataset')

fig.show()


# In[23]:


min_fares = train.groupby('Pclass')['Fare'].min().rename('Min Fare')
max_fares = train.groupby('Pclass')['Fare'].max().rename('Max Fare')
median_fares = train.groupby('Pclass')['Fare'].median().rename('Median Fare')
mean_fares = train.groupby('Pclass')['Fare'].mean().rename('Mean Fare')
std_fares = train.groupby('Pclass')['Fare'].std().rename('Std Fare')

fare_ranges = pd.concat([min_fares, max_fares, median_fares, mean_fares, std_fares], axis=1)

print(fare_ranges)


# In[24]:


# Replace 0 'Fare' value with median of corresponding Pclass
train['Fare'] = train.groupby('Pclass')['Fare'].transform(lambda x: x.replace(0, x.median()))


# In[25]:


min_fares = train.groupby('Pclass')['Fare'].min().rename('Min Fare')
max_fares = train.groupby('Pclass')['Fare'].max().rename('Max Fare')
median_fares = train.groupby('Pclass')['Fare'].median().rename('Median Fare')
mean_fares = train.groupby('Pclass')['Fare'].mean().rename('Mean Fare')
std_fares = train.groupby('Pclass')['Fare'].std().rename('Std Fare')

fare_ranges = pd.concat([min_fares, max_fares, median_fares, mean_fares, std_fares], axis=1)

print(fare_ranges)


# In[26]:


fig = px.box(train, x='Pclass', y='Fare', title='Distribution of Fare by Pclass')
fig.update_xaxes(title_text='Pclass')
fig.update_yaxes(title_text='Fare')
fig.show()


# In[27]:


def classify_fare_corrected(pclass, fare):
    median = fare_ranges.loc[pclass, 'Median Fare']
    if pclass == 1:
        std = fare_ranges.loc[pclass, 'Std Fare']
        q3 = median + 1.5 * std
        if fare < median:
            return 1  # Pclass 1 and below median
        elif fare <= q3:
            return 2  # Pclass 1 and above median but within IQR
        else:
            return 3  # Pclass 1 and above upper bound (IQR)
    elif pclass == 2:
        if fare <= median:
            return 4  # Pclass 2 and below median
        else:
            return 5  # Pclass 2 and above median
    else:  # Pclass 3
        if fare <= median:
            return 6  # Pclass 3 and below median
        else:
            return 7  # Pclass 3 and above median
        
train['Fare'] = train.apply(lambda row: classify_fare_corrected(row['Pclass'], row['Fare']), axis=1)


# In[28]:


# train['Fsize'] = train['SibSp'] + train['Parch']
train['Fsize'] = train['SibSp'] + train['Parch']
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[29]:


train.head()


# Drop 'PassengerId', 'Name', 'Ticket', 'Title' columns and get dummies for 'Sex', 'Embarked', 'Group'

# In[30]:


train.drop(['PassengerId', 'Name', 'Ticket', 'Title'], axis=1, inplace=True)


# In[31]:


# Get dummies for 'Sex' and 'Embarked'
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)


# In[32]:


train.head()


# In[33]:


train.isnull().sum()


# In[34]:


train.info()


# In[35]:


pip install -U imbalanced-learn


# # Train Test Split with SMOTE

# In[36]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = train.drop(['Survived'], axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Original Train Data - X shape:", X_train.shape, "y shape:", y_train.shape)
print("Test Data - X shape:", X_test.shape, "y shape:", y_test.shape)

target_counts = y_train.value_counts()
print("Class counts after SMOTE:\n", target_counts)


# In[37]:


class_distribution_df = pd.DataFrame({'Class': target_counts.index, 'Count': target_counts.values})

fig = px.bar(class_distribution_df, x='Class', y='Count', title='Class Distribution after SMOTE')
fig.update_xaxes(title_text='Class')
fig.update_yaxes(title_text='Count')
fig.show()


# # ML Models

# Lasso

# In[38]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Normalization - Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lasso_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

lasso_model.fit(X_train_scaled, y_train)

# Get feature names
feature_names = X.columns

# Get absolute coefficient values
feature_importance = np.abs(lasso_model.coef_)[0]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Feature Importance (Lasso Classification)")
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient Value")

plt.xticks(range(len(feature_importance)), feature_names, rotation=90)

plt.show()


# Ridge

# In[39]:


from sklearn.linear_model import Ridge

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

feature_names = X.columns

feature_importance = np.abs(ridge_model.coef_)

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Feature Importance (Ridge Regression)")
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient Value")

plt.xticks(range(len(feature_importance)), feature_names, rotation=90)

plt.show()


# Elastic Net

# In[40]:


from sklearn.linear_model import ElasticNet

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net_model.fit(X_train_scaled, y_train)

feature_names = X.columns

feature_importance = np.abs(elastic_net_model.coef_)

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title("Feature Importance (Elastic Net)")
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient Value")

plt.xticks(range(len(feature_importance)), feature_names, rotation=90)

plt.show()


# # Random Forest

# In[41]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'bootstrap': [True, False], 
    'max_features': ['log2','sqrt'], 
    'min_samples_leaf': [1, 2, 4], 
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_


# In[42]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


# In[43]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# XGBoost

# In[44]:


pip install xgboost


# In[45]:


from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],  
    'gamma': [0, 0.1, 0.2],         
    'alpha': [0, 0.1, 0.5],         
    'lambda': [1, 1.5, 2]           
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_xgb = grid_search.best_estimator_

y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[46]:


import joblib

joblib.dump(best_xgb, "xgb_classifier.joblib.dat")

y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# SVM

# In[47]:


from sklearn.svm import SVC

svm = SVC(random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_svm = grid_search.best_estimator_


# In[48]:


svm_classifier = SVC(C=1, gamma=0.1, kernel='rbf', random_state=42)

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_classifier.decision_function(X_test))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# # Neural Net_MLP

# In[49]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42, early_stopping=True)

param_grid = {
    'hidden_layer_sizes': [(100, 50), (50, 25), (200, 100)],
    'max_iter': [1000, 2000],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'validation_fraction': [0.1, 0.2],  
    'beta_1': [0.9, 0.95],              
    'beta_2': [0.999, 0.9999],          
    'n_iter_no_change': [10, 20]        
}

grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:")
print(grid_search.best_params_)


# In[50]:


best_mlp = grid_search.best_estimator_

y_pred = best_mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

y_prob = best_mlp.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("AUC-ROC Score:", roc_auc)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = [0, 1]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i][j]), horizontalalignment="center", color="white" if conf_matrix[i][j] > conf_matrix.max() / 2 else "black")
plt.tight_layout()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # Stacked Model (RF + MLP + XGB) + Meta Model: Logistic Regression

# In[51]:


from sklearn.ensemble import StackingClassifier

# Define the base models
base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('mlp',MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)),
    ('xgb', XGBClassifier(random_state=42))
]

# Define the meta-model
meta_model = LogisticRegression(random_state=42)

# Create the stacking model
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Define the parameter grid
param_grid = {
    # RandomForest parameters: Balance between performance and overfitting
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20],

    # MLP parameters: Layer sizes and regularization to control complexity
    'mlp__hidden_layer_sizes': [(100, 50), (50, 25)],
    'mlp__alpha': [0.0001, 0.001],

    # XGBoost parameters: Number of estimators and learning rate for performance tuning
    'xgb__n_estimators': [100, 200],
    'xgb__learning_rate': [0.01, 0.1]
}

# Create the grid search
grid_search = GridSearchCV(estimator=stacked_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

best_stacked = grid_search.best_estimator_

y_pred = best_stacked.predict(X_test)
y_proba = best_stacked.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# # Model Selection and Prediction

# Best ML model is...
# Stacked Model (RF + XGB + MLP) + Meta Model: Logistic Regression
# 
# Accuracy: 0.7921348314606742
# 
# Precision: 0.7424242424242424
# 
# Recall: 0.7101449275362319
# 
# F1 Score: 0.725925925925926
# 
# 0.86198643797367379736737 67370858

# In[52]:


joblib.dump(stacked_model, "stacking_classifier.joblib.dat")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[53]:


train.head()


# In[54]:


test.head()


# In[55]:


test.info()


# In[56]:


test['Age'] = test['Age'].astype(float)

test['Age'].fillna(28, inplace=True)

bins_4 = [0, 13, 18, 65, 81]
labels_4 = [1, 2, 3, 4]

test['Age'] = pd.cut(test['Age'], bins=bins_4, labels=labels_4, right=False)
test['Age'] = test['Age'].astype(int)


# In[57]:


def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

test['Title'] = test['Name'].apply(extract_title)

title_counts = test['Title'].value_counts()

print(title_counts)

def map_title_to_group(title):
    if title in ['Mr', 'Miss', 'Mrs', 'Ms']:
        return 1
    elif title in ['Don', 'Sir', 'Countess', 'Lady', 'Jonkheer']:
        return 2
    elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
        return 3
    elif title in ['Master', 'Mlle', 'Mme']:
        return 4
    else:
        return 5

test['Group'] = test['Title'].apply(map_title_to_group)


# In[58]:


# Replace 0 'Fare' value with median of corresponding Pclass
test['Fare'] = test.groupby('Pclass')['Fare'].transform(lambda x: x.replace(0, x.median()))


# In[59]:


def classify_fare_corrected(pclass, fare):
    median = fare_ranges.loc[pclass, 'Median Fare']
    if pclass == 1:
        std = fare_ranges.loc[pclass, 'Std Fare']
        q3 = median + 1.5 * std
        if fare < median:
            return 1  # Pclass 1 and below median
        elif fare <= q3:
            return 2  # Pclass 1 and above median but within IQR
        else:
            return 3  # Pclass 1 and above upper bound (IQR)
    elif pclass == 2:
        if fare <= median:
            return 4  # Pclass 2 and below median
        else:
            return 5  # Pclass 2 and above median
    else:  # Pclass 3
        if fare <= median:
            return 6  # Pclass 3 and below median
        else:
            return 7  # Pclass 3 and above median
        
test['Fare'] = test.apply(lambda row: classify_fare_corrected(row['Pclass'], row['Fare']), axis=1)


# test['Fsize'] = test['SibSp'] + test['Parch']

# In[60]:


# test['Fsize'] = test['SibSp'] + test['Parch']
test['Fsize'] = test['SibSp'] + test['Parch']
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[61]:


test.drop(['PassengerId', 'Name', 'Ticket', 'Title','Cabin'], axis=1, inplace=True)


# In[62]:


# Get dummies for 'Sex' and 'Embarked'
test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)


# In[63]:


test.head()


# In[64]:


test.info()


# In[65]:


X_train.head()


# In[66]:


best_stacked = joblib.load("stacking_classifier.joblib.dat")

best_stacked.fit(X_train, y_train)

y_pred = best_stacked.predict(test)


# In[67]:


test['Survived'] = y_pred

test.to_csv('test_with_predictions.csv', index=False)


# # Prediction

# In[68]:


pd.set_option('display.max_rows', 500)
test.head(419)

