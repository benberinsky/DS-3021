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
y= train_data['A_HGA']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=21)
X_tune, X_test, y_tune, y_test = train_test_split(X_test,y_test,  train_size = 0.50, random_state=49)

kf = RepeatedStratifiedKFold(n_splits=10,n_repeats =5, random_state=42)

param={"max_depth" : [2,4,6,8,10]}

cl= DecisionTreeClassifier(random_state=672)

scoring = ['roc_auc_ovr', 'recall_macro', 'balanced_accuracy']

search = GridSearchCV(cl, param, scoring=scoring, n_jobs=-1, cv=kf,refit='balanced_accuracy')

model = search.fit(X_train, y_train)

# %%
best = model.best_estimator_
print(best)

# %%
print(model.cv_results_.keys())


# %%
bal_acc= model.cv_results_['mean_test_balanced_accuracy']

#Parameter:
depth= np.unique(model.cv_results_['param_max_depth']).data

#Build DataFrame:
final_model = pd.DataFrame(list(zip(depth, bal_acc)),
               columns =['depth', 'bal_acc'])

#Let's take a look
final_model.style.hide(axis='index')


