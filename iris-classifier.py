from sklearn import datasets
iris= datasets.load_iris()

X= iris.data
y= iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

#Training the classifier
my_classifier.fit(X_train, y_train)

#prediction the results
prediction= my_classifier.predict(X_test)


print("prediction %s",prediction)
print("test %s",y_test)

#evaluation the algorithm
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction))