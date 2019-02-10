#Iris
#decision tree
#flowers
#https://www.youtube.com/watch?v=tNa99PG8hR8&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=2

#sklearns includes the dataset of iris
import numpy as np
from sklearn import tree 
from sklearn.datasets import load_iris
iris= load_iris()
#id of flowers that we are going to test:
test_idx=[0,50,100]

#making the train data base without the testing flowers:
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#test features flowers
test_target =iris.target[test_idx]
test_data = iris.data[test_idx]

clf =tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)


# print features and labels which to classify
print(iris.feature_names)
print(iris.target_names)
"""
print("features %s" %(iris.data[0]))
print(iris.target[0])
"""

print(test_target)
print(clf.predict(test_data))

#visualization of the training tree creating a pdf, 
#install pydotplus first (conda install -c conda-forge pydotplus)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")  