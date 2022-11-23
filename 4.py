from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression())
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
