import bentoml

from sklearn import svm
from sklearn import datasets
#LOAD TRAINING DATASET

iris = datasets.load_iris()
X, y = iris.data, iris.target

#train model

clf = svm.SVC(gamma= 'scale')
clf.fit(X,y)

#save the model to bentoml local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f'Model saved: {saved_model}')