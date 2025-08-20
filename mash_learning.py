from sklearn.tree import DecisionTreeClassifier
x_train = [[1],[2],[3],[4],[5],[6],[7]]
y_train = [0, 0, 0, 1, 1, 1, 0]

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

x_new = [[3], [5.5],[7]]
y_new = [0,1,1]

predictions = clf.predict(x_new)

print("New data:", x_new)
print("Predictions", predictions)
print("Actual labels:", y_new)
print("Accuracy on new data:", clf.score(x_new,y_new))

from sklearn import tree
import  matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
tree.plot_tree(clf, feature_names=["Hours"], class_names=["Fail", "Pass"], filled=True)
plt.show()

plt.figure(figsize=(12,3))
tree.plot_tree(clf, feature_names=["Hours"], class_names=["Fail", "Pass"], filled=True)
plt.show()

from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40, 50, 60, 70, 80]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6.1]])
print("Predicted score for 6 hours studed:", prediction[0])

print("R^2 score:", model.score(X, y))
import matplotlib.pyplot as plt
plt.scatter(X, y, color="blue", label ="Data points")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.scatter(6, prediction, color="green", s=100, marker="*", label="Prediction (6h)")

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.legend()
plt.show()