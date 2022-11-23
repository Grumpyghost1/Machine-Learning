from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
 
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf=MLPClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
 
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
