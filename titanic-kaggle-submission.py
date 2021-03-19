# achieves 77.27% on Kaggle Titanic Dataset (only using columns: 'age', 'sex'(dummy), and 'fare')
# result with sk-learn is 72.7%

import pandas as pd
import numpy as np
import sys 
sys.path.append(".")

from naiveBayes import MultinomialNaiveBayes

# LOAD TRAINING DATA =================================================================================

df = pd.read_csv('trainTitanic.csv')

# drop unnecessary columns 
df = df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# turn male female into dummies and drop old male/female column 
dummies = pd.get_dummies(df['Sex'])
df = df.drop(['Sex'], axis=1)

# add dummies 
df= pd.concat([df, dummies], axis=1)

# turn survived column into separate y, then delete
y = df['Survived']
df = df.drop(['Survived'], axis=1)

# fill any NaN values with the mean of the column
df = df.fillna(df.mean())

# convert both to numpy arrays to pass to MultnomialNaiveBayes
A = df.to_numpy()
y = y.to_numpy().reshape((-1, 1))


# LOAD TEST DATA =======================================================================================
df2 = pd.read_csv('testTitanic.csv')
passengerNos = df2['PassengerId']

# drop unnecessary columns 
df2 = df2.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# turn male female into dummies and drop old male/female column 
dummies = pd.get_dummies(df2['Sex'])
df2 = df2.drop(['Sex'], axis=1)

# add dummies 
df2 = pd.concat([df2, dummies], axis=1)
X = df2.to_numpy()

# IMPLEMENT NAIVE BAYES ================================================================================

# pass training data into MultinomialNaiveBayes
MNB = MultinomialNaiveBayes(A, y)

# fit new examples (test set)
yhat = MNB.fit(X)

# create submission csv file
submissionDf = np.concatenate([passengerNos.to_numpy().reshape((-1, 1)), yhat], axis=1)
submission = pd.DataFrame(data=submissionDf, columns=["PassengerId", "Survived"])
submission.to_csv('TitanicSubmission1.csv', index=False)
