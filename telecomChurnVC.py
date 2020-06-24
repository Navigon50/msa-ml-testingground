# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Preliminary Data Preprocessing

#Import Required Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from pylab import rcParams
from heatmap import heatmap, corrplot
from sklearn.feature_selection import *
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# +
# Import dataset from csv
data = pd.read_csv("telcoCustomerChurn.csv")

# Print first 10 rows to have an elementary view of the data
data.head(5)
#data.info()
# -

# From the data above we can see while the dataset looks functional, there appear to be some basic errors with the input of the data and column names. Specifically
# * All column names are capitalized, except for tenure, and totalcharges, as well as a few others
# * totalCharges is a python object variable, rather than a floating 64 point number as it should represent a total dollar figure.
# * SeniorCitizen is capitalized, and is characterized as numeric data, when it is more likely to be boolean or factor based data.
# Hence, we reread the data, renaming and assigning new data types as required using a python dictionary and the secondary arguments of the read_csv function

# +
# Define the dictionary pairings
nameDict= {'customerID':'object','gender':'object','seniorCitizen':'object','partner':'object','dependents': 'object' ,'tenure':np.int64,'phoneService':'object','multipleLines':'object','internetService':'object','onlineSecurity':'object','onlineBackup':'object','deviceProtection':'object','techSupport':'object','streamingTV':'object','streamingMovies':'object','contract':'object','paperlessBilling':'object','paymentMethod':'object','monthlyCharges':np.float64,'totalCharges':np.float64,'churn':'object'}

# Reread the csv wth added changes.
df = pd.read_csv("telcoCustomerChurn.csv", header = 0, names = ['customerID','gender','seniorCitizen','partner','dependents','tenure','phoneService','multipleLines','internetService','onlineSecurity','onlineBackup','deviceProtection','techSupport','streamingTV','streamingMovies','contract','paperlessBilling','paymentMethod','monthlyCharges','totalCharges','churn'], na_values = " ",dtype=nameDict)

#  Examine the first five rows
df.head(5)
# -

# Examine the data types of the changed csv
df.info()

# Now that basic data types are fixed, we want to go the usual route and check for any missing data, duplicates or strange results.

# Check for Null Values
df.isnull().values.any()
print(df.isnull().sum())

# Evidently there are missing values in the totalCharges column, which do not appear to be clear as to why they are missing values. Lookning back at the original CSV, for the 11 missing values those contracts have 0 in their tenure, most likely having cancelled their contract before the month was up. This obviously meant that they weren't charged anyting in total, so this missing values should really just be zeros.

# +
# Replace the missing values in totalCharges for the rows in the dataframe which have 0 in tenure
df[df['tenure']==0]=df[df['tenure']==0].replace(np.nan,0)

# Now after running this command let us check for missing values:
print(df.isnull().sum())

# +
# Next, we check for any duplicated values, fortunately there is none, so we can move on.

# Check for duplicated values
df.duplicated().values.sum()
# -

# Drop customerID from the dataframe as while it provides useful individual information
# it is not helpful in aggregate, due to the large variety of unique values.
# Moreoever, we also drop totalCharges to avoid the issue of collinearity with monthly charges present.
df = df.drop(['customerID','totalCharges'], axis=1)

# # Exploratory Data Analysis

# Let us examine the numerical variables statistical summary to see if we can find anything out of the ordinary.
df.describe(include = ['float64','int64'])

# +
# Caclualte coefficients of variations

df.select_dtypes(include=['float64','int64'])
cv =  lambda x: np.std(x) / np.mean(x)
var = np.apply_along_axis(cv, axis=0, arr=df.select_dtypes(include=['float64','int64']))
print(var)
# -

df.info()

ax = sns.boxenplot(x="tenure", y="churn", data=df)

sns.catplot(x="contract", hue="churn",data=df, kind="count",height=4, aspect=1);

sns.catplot(x="tenure", y="contract",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="monthlyCharges", y="contract",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="tenure", y="techSupport",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="monthlyCharges", y="churn",
            row="techSupport", aspect=.6,
            kind="box", data=df);


sns.catplot(x="tenure", y="onlineSecurity",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="monthlyCharges", y="onlineSecurity",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="tenure", y="internetService",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="monthlyCharges", y="internetService",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="tenure", y="paymentMethod",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

sns.catplot(x="monthlyCharges", y="paymentMethod",

                hue="churn",

                data=df, kind="box",

                height=6, aspect=1);

# # Model Preprocessing

# As the majority of features used in the dataset are primarily categorical features, converting them to dummy variables via oneHotEncoding and OrdinalEncoder methods from scikitlearn will enable us to utilize classification machine learning methods on the data.

# Get Dummies
dfdum=pd.get_dummies(df,prefix_sep='_')

# Tenure has a very high coefficient of variation compared to monthlyCharges, indicating it's much higher variability. 
# Hence, before we integrate this into our model framework, we should standardize them to be on the same scale.

# Calculate correlation data for each column pairing in the dataset (numerical only)
# Rebalancing unbalanced classes using SMOTE resampling, by first importing relevant packages
# Install a pip package in the current Jupyter kernel
#import sys
#!{sys.executable} -m pip install heatmapz


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# Analysing the correlation plot, we see most features are fairly weakly correlated, so we want to explore features that do have a high
# correlation with churning
# From the graph, these appear to be, tenure (which is negative correlated, month to month contracts, presence of tech support, online security, fiber optic service, and electronic check payment methods
# fiber optic is also strongly corelated with monthyl charges, especially so if no internet service was present. 

c = df.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
so[so <1]

# # Feature Selection

df.shape

#Split the dataset into features and target variable
X = dfdum.iloc[:, 0:45]
y = dfdum.iloc[:,46:47]


type(y)

selector = SelectPercentile(mutual_info_classif, percentile=10)
selector.fit(X, y.values.ravel())
# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
X_new = X.iloc[:,cols]
X_new.columns


# Under chi2 tests, the 10 best features are 'tenure', 'monthlyCharges', 'internetService_Fiber optic',
#        'onlineSecurity_No', 'techSupport_No',
#        'streamingTV_No internet service',
#        'streamingMovies_No internet service', 'contract_Month-to-month',
#        'contract_Two year', 'paymentMethod_Electronic check'
# How about under F, tests?
# 'tenure', 'internetService_Fiber optic', 'onlineSecurity_No',
#        'onlineBackup_No', 'deviceProtection_No', 'techSupport_No',
#        'streamingMovies_No internet service', 'contract_Month-to-month',
#        'contract_Two year', 'paymentMethod_Electronic check']
# Roughly the same, so what happens when we consider mutual info classification which looks at non-linear dependencies?
# ['tenure', 'monthlyCharges', 'internetService_Fiber optic',
#        'onlineSecurity_No', 'onlineBackup_No', 'techSupport_No',
#        'streamingTV_No internet service', 'contract_Month-to-month',
#        'contract_Two year', 'paymentMethod_Electronic check']
# Top 10% of featurse:
# ['tenure', 'internetService_Fiber optic', 'onlineSecurity_No',
#        'techSupport_No', 'contract_Month-to-month'],
#       dtype='object')
# Top 10% Non-linear features tenure', 'onlineSecurity_No', 'techSupport_No',
#        'contract_Month-to-month', 'paymentMethod_Electronic check'
#        
# Recursive Feature Selection
#

# +
# Recursive Feature Elimination:
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y.values.ravel())

print("Optimal number of features : %d" % rfecv.n_features_)

## Optimal number of features is 42
# -

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# +
#L1 Based Feature Selection
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter = 5000).fit(X, y.values.ravel())
model = SelectFromModel(lsvc, prefit=True)
X_svc = model.transform(X)
cols = model.get_support(indices=True)
X_svc = X.iloc[:,cols]
X_svc.columns

# results might be a bit dodgy
# -

# Tree Based Feature Selection
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y.values.ravel())
clf.feature_importances_
modeltree = SelectFromModel(clf, prefit=True)
X_tree = modeltree.transform(X)
cols = modeltree.get_support(indices=True)
X_svc = X.iloc[:,cols]
X_svc.columns


# # Removing Collinearity

# +
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define a function to calculate the respective VIF for each variable in the daaset
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif) 


# -

# Calculate the VIF for the dataset
calc_vif(df)

# # Experimenting with Classification Algorithims

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train.info()

# +
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(solver = 'saga', random_state=0, max_iter = 2000)
logr.fit(X_train, y_train.values.ravel())
y_pred=logr.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logr.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logr_roc_auc = roc_auc_score(y_test, logr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logr_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# -

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
pca = PCA()
# set the tolerance to a large value to make the example faster
logistic = LogisticRegression(penalty = 'l2', solver = 'sag',max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[('smt', smt),('fs', SelectKBest(f_classif)), ('logistic', logistic)])
# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'fs__k': [5, 15, 30, 42,45],
    'logistic__C': np.logspace(-0.01, 0.01, 20),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1,scoring = "recall")
search.fit(X_train, y_train.values.ravel())
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)Best parameter (CV score=0.853):
#{'fs__k': 5, 'logistic__C': 1.0208157648868061}
# bst is sad 0.846


# Linear Discriminant Analysis, LDA same performance as logistic, QDA worse performance because of collinearity
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train.values.ravel())
y_predqda=qda.predict(X_test)
print('Accuracy of QDA classifier on test set: {:.2f}'.format(qda.score(X_test, y_test)))
print(classification_report(y_test, y_predqda))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())
y_predgnb=gnb.predict(X_test)
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))
print(classification_report(y_test, y_predgnb))


# +
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Make a pipelien and scale the input.
sgd = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
sgd.fit(X_train, y_train.values.ravel())

# Calculate probabilities, accuracy, recall rates.
y_predsgd=sgd.predict(X_test)
print('Accuracy of SGD classifier on test set: {:.2f}'.format(sgd.score(X_test, y_test)))
print(classification_report(y_test, y_predsgd))


# +
# Decision Tree Classifier
from sklearn import tree
dtreec = tree.DecisionTreeClassifier()
dtreec.fit(X_train, y_train.values.ravel())

# Calculate probabilities, accuracy, recall rates.
y_predtree=dtreec.predict(X_test)
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(dtreec.score(X_test, y_test)))
print(classification_report(y_test, y_predtree))
# -

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, random_state=42)
rf.fit(X_train, y_train.values.ravel())
# Calculate probabilities, accuracy, recall rates.
y_predrf=rf.predict(X_test)
print('Accuracy of RandomForest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
print(classification_report(y_test, y_predrf))

# +
# AdaBoost
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train,y_train.values.ravel())

# Calculate probabilities, accuracy, recall rates.
y_predada=ada.predict(X_test)
print('Accuracy of AdaBoost classifier on test set: {:.2f}'.format(ada.score(X_test, y_test)))
print(classification_report(y_test, y_predada))

scores = cross_val_score(ada, X_test, y_test.values.ravel(), cv=5)
scores.mean()

# +
# Baggingfrom sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
bagc = BaggingClassifier(base_estimator=LogisticRegression(solver = 'sag',max_iter=10000, tol=0.1), random_state=42)
bagc.fit(X_train,y_train.values.ravel())

# Calculate probabilities, accuracy, recall rates.
y_predbag=bagc.predict(X_test)
print('Accuracy of Bagging classifier on test set: {:.2f}'.format(bagc.score(X_test, y_test)))
print(classification_report(y_test, y_predbag))

# +
# NewogRes
logis=LogisticRegression(penalty = 'l2', solver = 'sag',max_iter=10000, tol=0.1, C = 1.02)

# New bagc
baglog= BaggingClassifier(base_estimator=logis, random_state=42, n_estimators = 60, max_samples = 6, max_features = 2)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    #'n_estimators':range(20,100,10)
    #'max_samples':range(1,100,5),
    #'min_samples_split':range(200,2000,200),
    #'min_samples_leaf':range(1,25,1)
    'max_features':range(1,45,1)
    #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
    #'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,1.0]
    
}
search = GridSearchCV(baglog, param_grid, n_jobs=-1,scoring = "recall", cv =5)
search.fit(X_train_res, y_train_res.values.ravel())
print("Best parameter (CV score=%0.3f):" % search.best_score_)
search.best_params_, search.best_score_

# n_estimators 60 (95% ACCURACY...)
# -

bagc.get_params().keys()

# +
#Gradient Boostd Decision Trees (Maybe have a look at SVC later.)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate = 0.04,n_estimators = 550,max_features='sqrt',subsample=0.8,random_state=42)
gbc.fit(X_train,y_train.values.ravel())

# Calculate probabilities, accuracy, recall rates.
y_predgb=gbc.predict(X_test)
print('Accuracy of Gradient Boosted classifier on test set: {:.2f}'.format(gbc.score(X_test, y_test)))
print(classification_report(y_test, y_predgb))
# -

gbc.get_params().keys()

sm = SMOTENC(random_state=42, categorical_features=[18, 19]) 
X_res, y_res = sm.fit_resample(X, y)
gbc =GradientBoostingClassifier(learning_rate=0.05,min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=42, n_estimators=220)
pipe = Pipeline(steps=[('sm', sm),('gbc', gbc)])

 X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, random_state=42)



# +
# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
   # 'n_estimators':range(20,1000,10)
    #'max_depth':range(1,25,2),
    #'min_samples_split':range(200,2000,200),
    #'min_samples_leaf':range(1,25,1)
    #'max_features':range(1,45,1)
    #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
    #'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,1.0]
    
}
search = GridSearchCV(gbc, param_grid, n_jobs=-1,scoring = "recall", cv =5)
search.fit(X_train_res, y_train_res.values.ravel())
print("Best parameter (CV score=%0.3f):" % search.best_score_)
search.best_params_, search.best_score_

# {'max_depth': 3, 'min_samples_split': 1200} n_estimators 550 .863 acc min samples lef 23, max_features = 1, subsample 0.8

# -

 search.best_params_, search.best_score_



# +
# XG Boost
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

import xgboost as xgb
model=xgb.XGBClassifier(random_state=42,learning_rate=0.01)
model.fit(X_train, y_train.values.ravel())
model.score(X_test,y_test)

# +
# Cat Boost
#from catboost import CatBoostClassifier
#model=CatBoostClassifier() 
# Try and get this to work later

#model.fit(xtrain,ytrain,[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16],eval_set=(xtest, ytest))
#model.score(x_test,y_test)
# -

# # Unbalanced Classes
# One major issue often found in many classification problems is the problem of unbalanced classes. This refers to the fact that for classification problems, the majority class generally has more samples or exists in greater proportions than the minority class, which skews the classification algorithim's predictive capacity. 
#
# In layman's terms if there are 900 nos and 100 yes in a classification dataset, the overwhelming majority of the no's are going to make the model very good at predicting a no case, but not so good when predicting a yes case.
#
# We can counter these effects by considering resmapling strategies, and in particular I will be considering an adjustment to the SMOTE class of resampling strategies which incorporates categorical data. 

# Check class totals to examine evidence for imbalanced classes
pd.crosstab(df.churn,df.churn, normalize = True)

# As we can see from above, there is a definite imbalance in the number of customers who did not churn, which means that classification algorithims are likely to be significantly more biased towards predicting no, as to predicting yes
#
# To simplify the rerunning of previous algorithims under a SMOTE-resampling outlook, we shall take advantage of python's pipeline feature, to also perform feature selection to improve accuracy.

# +
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# Implement the required elements
pca = PCA()
smt = SMOTENC(categorical_features=range(2,45), random_state=42)
classifiers = [
    LogisticRegression(solver = 'saga', random_state=42, max_iter = 10000),
    GaussianNB(),
    tree.DecisionTreeClassifier(),
    RandomForestClassifier(max_depth=2, random_state=42),
    BaggingClassifier(base_estimator=SVC(), n_estimators=20, random_state=42),
    AdaBoostClassifier(n_estimators=200),
    GradientBoostingClassifier(random_state=42),
    xgb.XGBClassifier(random_state=42,learning_rate=0.01)
    ]

for classifier in classifiers:
    pipe = Pipeline(steps=[('scale',StandardScaler()),('smt', smt),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train.values.ravel())   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))
    


# Accuracy went down to 77% after resampling, but precision is quite high.
# Naive Bayes holds similiar statistics, with a accuracy of 71%
# Decision Tree had accuracy of 72%, random forests 74%
# Bagging: 74% accuracy, similiar stats to log
# Gradient Boosted 76% (second highest), tied with ada.
# Seems clear that a mixutre of logr, gdb + ada would be useful

# now, pipeline with feature selection via PCA 


# -

# Pipeline with Feature Selection
#pipeFS = Pipeline([
 # ('feature_selection', pca),
  #  ('smt',smt),
  #('classification', LogisticRegression(solver = 'saga', random_state=42, max_iter = 10000))
#])
#pipeFS.fit(X_train, y_train.values.ravel())
print("model score: %.3f" % pipeFS.score(X_test, y_test))



# +
# Pipeline with GridSearch

# +
# Combine for final model pipeline
