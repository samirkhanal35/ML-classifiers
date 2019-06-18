from pandas import read_csv #to read csv file, csv -> comma separated values
from numpy import set_printoptions #to print precision in accuracy
from sklearn.model_selection import train_test_split #to split train and test from a dataset


from sklearn.svm import SVC #importing Support Vector Classifier

filename = 'dataset.csv' #dataset.csv -> name of dataset
dataframe = read_csv(filename) 

array = dataframe.values
X = array[:,1:] #all from column 1
Y = array[:,0] #column 0 of column labels

test_size = .30 #30% of dataset to test set and remaining to training set
seed = 45 #45 random states , you can use any number of random states 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = SVC()
model.fit(X_train, Y_train) #fitting of the model

result = model.score(X_test, Y_test) #result of testing of the model 

print("Accuracy: %.3f%%") % (result*100.0) #printing the accuracy of the model
