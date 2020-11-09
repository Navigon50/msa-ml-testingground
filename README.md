# Telecom Churn project

## Idea/Project
The key idea for the project was to explore and test my knowledge of the machine learning syntax and code available in python, after having learned most of my machine learning methods from R. However I also wanted to explore a particular project somewhat relevant to my domain area, which is why I examined the telecomChurn open data project on kaggle.

As an actuarial student, we are often involved with the calculation and quantification of long-term uncertain cashflows, especially for businesses that hold long-term contracts with their customers, such as a mobile phone contract. Therefore, I wanted to build a model that would balance predictability and precision to help me understand the nature of the dataset.

However, my coursework ended up taking priority over the project so I decided to stop with just building a proper inferential model rather than exploring the model with the best possible performance.

# Environment Setup and underlying dependencies
Note that I use anaconda as my main application for managing my python workflow, so packages can be installed using conda or pip.

```python 

#Import Required Packages for data analysis and manipulation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from heatmap import heatmap, corrplot
from sklearn.feature_selection import *

# Import Models 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

# Import Model Selection Tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# import metrics and other useful tools
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import PCA for Dimensionality Reduction
from sklearn.decomposition import PCA

# Import packages for over-sampling
from collections import Counter
from imblearn.over_sampling import SMOTENC

```

# Training and Testing Model
The entire process for training and testing the model is outlined in the python notebook, which I will reproduce here summarized for ease of view.

## Importing Data and preprocessing
```python
# Reread the csv wth added changes.
df = pd.read_csv("telcoCustomerChurn.csv", header = 0, names = ['customerID','gender','seniorCitizen','partner','dependents','tenure','phoneService','multipleLines','internetService','onlineSecurity','onlineBackup','deviceProtection','techSupport','streamingTV','streamingMovies','contract','paperlessBilling','paymentMethod','monthlyCharges','totalCharges','churn'], na_values = " ",dtype=nameDict)

# Check for Null Values
df.isnull().values.any()
print(df.isnull().sum())

# Replace the missing values in totalCharges for the rows in the dataframe which have 0 in tenure
df[df['tenure']==0]=df[df['tenure']==0].replace(np.nan,0)

# Now after running this command let us check for missing values:
print(df.isnull().sum())

# Drop customerID from the dataframe as while it provides useful individual information
# it is not helpful in aggregate, due to the large variety of unique values.
# Moreoever, we also drop totalCharges to avoid the issue of collinearity with monthly charges present.
df = df.drop(['customerID','totalCharges'], axis=1)

# Get Dummies
dfdum=pd.get_dummies(df,prefix_sep='_')

```

## Training and Testing
```python
#Split the dataset into features and target variable, namely X and y.
X = dfdum.iloc[:, 0:45]
y = dfdum.iloc[:,46:47]

### Resampling 

# We use SMOTENC here as it is SMOTE resampling that accounts for categorical features
# Create the rules for resampling
sm = SMOTENC(random_state=42, categorical_features=[18, 19]) 

# Resample our given dataset
X_res, y_res = sm.fit_resample(X, y)

# Create new train test splits based on this dataset.
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_res, y_res, random_state=42)

# Define a new Gradient Boosting Classifier and metrics to be used in hyperparameter testing.
gbc =GradientBoostingClassifier(n_estimators = 770, learning_rate=0.05,max_features=10,subsample=0.8,random_state=42, max_depth = 3, min_samples_split = 400)
scoring = {'AUC': 'roc_auc', 'f1': 'f1'}

# Create a parameter grid for all the parameters you wish to tune, starting with learning rate then progressing 
# onto n_estimators, max depth and so on.
param_grid = {
   #'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,1.0]
   #'n_estimators':range(20,1000,10)
   #'max_depth':range(1,25,2),
   #'min_samples_split':range(200,2000,200),
   # 'min_samples_leaf':range(1,25,1)
   # 'max_features':range(1,45,1)
   #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]    
}

# Create a new GridSearch object to perform extensive hyperparameter scanning for the above model
# using all cores of the cpu and focusing paritcularly on the AUC metric.
# cross validate your scores using 5-fold cross validation
gs = GridSearchCV(gbc, param_grid, n_jobs=-1,scoring = scoring,refit='AUC', cv = 5)

# Fit the new GridSearch Object onto your resampled dataset.
gs.fit(X_train_res, y_train_res.values.ravel())

# Print the best scores and best parameters.
print("Best parameter (CV score=%0.3f):" % gs.best_score_)
gs.best_params_, gs.best_score_
```
Note that during hyperparameter tuning, the vairables are uncommented one at a time and the model successively rerun to obtain their tuned values.

## Final Model
```python
# Build the final model using the tuned parameters from before
bestgbc =GradientBoostingClassifier(n_estimators = 770, learning_rate=0.05,max_features=10,subsample=0.8,random_state=42, max_depth = 3, min_samples_split = 400)

# Put together the final pipeline with scaled inputs for the model, and make predictions.
finalpipe = Pipeline(steps=[('scale',StandardScaler()),('classifier', bestgbc)])
finalpipe.fit(X_train_res, y_train_res.values.ravel())
y_predfinal=finalpipe.predict(X_test_res)
y_predprobs=finalpipe.predict_proba(X_test_res)
```
