#Hello World: Procedimientos de aprendizaje automático #1
#https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels=[0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150,0]]))