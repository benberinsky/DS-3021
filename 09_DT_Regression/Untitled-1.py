# %%
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import make_scorer, roc_auc_score


from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz 

# %%
train_data = pd.read_csv("../data/CENSUS_ED_ATTN.csv")
test_data = pd.read_csv("../data/Census_Test.csv")

# %%
train_data.info()

# %%
train_data.isna().sum()

# %%
print(train_data['A_HGA'].value_counts())

# %%
print(train_data['PENATVTY'].value_counts())

# %%
# Dropping military service
train_data = train_data.drop('PEAFEVER', axis=1)
train_data = train_data.drop('PEHSPNON', axis=1)

# %%
# Collapsing country to immigrant vs. from U.S
train_data['PENATVTY'] = train_data['PENATVTY'].apply(lambda x: 0 if x == 57 else 1)
train_data['PEFNTVTY'] = train_data['PEFNTVTY'].apply(lambda x: 0 if x == 57 else 1)


# %%
train_data['PRDTRACE'] = train_data['PRDTRACE'].apply(lambda x: 0 if x == 1 else 1)

# %%
ordinal_list = ['A_SEX', 'PARENT', 'PENATVTY', 'PEFNTVTY', 'PEINUSYR', 'PEPAR1TYP', 'PRCITSHP', 'PRDTRACE', 'ERN_SRCE']
ordinal_encoder = OrdinalEncoder()
for i in ordinal_list:
    train_data[[i]] = ordinal_encoder.fit_transform(train_data[[i]])

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train_data['A_HGA'])


# %%
X= train_data.drop(columns='A_HGA')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, stratify = y, random_state=21)
X_tune, X_test, y_tune, y_test = train_test_split(X_test,y_test,  train_size = 0.50, stratify = y_test, random_state=49)

kf = RepeatedStratifiedKFold(n_splits=10,n_repeats =5, random_state=42)

param = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cl= DecisionTreeClassifier(random_state=672)

scoring = ['roc_auc_ovr', 'precision_macro', 'balanced_accuracy']

search = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='precision_macro')

model = search.fit(X_train, y_train)

# %%
best = model.best_estimator_
print(best)

# %%
bal_acc= model.cv_results_['mean_test_balanced_accuracy']
roc_auc = model.cv_results_['mean_test_roc_auc_ovr']
recall = model.cv_results_['mean_test_recall_macro']


#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model = pd.DataFrame(list(zip(depth, roc_auc, bal_acc, recall)),
               columns =['depth', 'roc_auc', 'bal_acc', 'recall'])

#Let's take a look
final_model.style.hide(axis='index')

# %%
print(model.cv_results_.keys())


# %%
# Get best model results
print("Best parameters:", model.best_params_)
print("Best cross-validation score:", model.best_score_)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model.best_estimator_, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()

# %%
bal_acc= model.cv_results_['mean_test_balanced_accuracy']

#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model = pd.DataFrame(list(zip(depth, bal_acc)),
               columns =['depth', 'bal_acc'])

#Let's take a look
final_model.style.hide(axis='index')

# %%
# Ensure that y and y_pred have the same number of samples
y = y[:len(y_pred)]

# Calculate the classification report
report = classification_report(y, y_pred, output_dict=True)
macro_precision = report['macro avg']['precision']
print("Macro Precision:", macro_precision)

# %%
param = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = GridSearchCV(cl, param, scoring='precision_macro', cv=kf)
model = search.fit(X_train, y_train)


