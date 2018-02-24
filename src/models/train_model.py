import numpy as np
import pandas as pd
from models import nnutil

# Read the X_train_orig and create a matrix
train = pd.read_csv('../../data/processed/train-clean.csv')
X = train[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Title', 'FSize']].as_matrix().T
X_train = X[:, 0:850]
X_cv = X[:, 850:891]
#assert (np.shape(X_train) == (7, 891))

Y = train['Survived'].as_matrix().reshape((891, 1)).T
Y_train = Y[:, 0:850]
Y_cv = Y[:, 850:891]
#assert (np.shape(Y_train) == (1, 891))

# Define NN(include input layer)
layers_dim = np.array([7, 5, 3, 1])

# Training the model
np.random.seed(5)
parameters = nnutil.nn_model(X_train, Y_train, layers_dim, lambd=0.01, num_itr=35000, print_cost=True)
predictions = nnutil.predict(X_cv, parameters)
nnutil.eval_model(Y_cv, predictions)
# Prediction and file creationprint("The precision is " + str(precision))

test = pd.read_csv('../../data/processed/test-clean.csv')
X_test = test[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Title', 'FSize']].as_matrix().T
Y_test = nnutil.predict(X_test, parameters).T
test['Survived'] = Y_test
final = test[['PassengerId', 'Survived']]
final.to_csv('../../data/processed/result.csv', index=False)











