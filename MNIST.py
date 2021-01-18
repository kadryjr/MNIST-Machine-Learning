import matplotlib.pyplot as plt
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def applyHog(x):
    list = []
    for i in x:
        img = i.reshape(28, 28)
        features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        list.append(features)
    return list


# splitingdata
x_train, y_train = loadlocal_mnist(
    images_path='train-images.idx3-ubyte',
    labels_path='train-labels.idx1-ubyte'
)

x_test, y_test = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte',
    labels_path='t10k-labels.idx1-ubyte'
)

#x_train = applyHog(x_train)
#x_test = applyHog(x_test)
#print(x_train)
#print(x_test)


# feautres scalling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# KNN Classifier
k = 9
classifier1 = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
classifier1.fit(x_train, y_train)

# predict
y_pred1 = classifier1.predict(x_test)

# confuion mat
cm1 = confusion_matrix(y_test, y_pred1)

print("KNN classifier accuracy :", metrics.accuracy_score(y_test, y_pred1))

print(classification_report(y_pred1, y_test))



# Naive Bayes Classifier
classifier2 = GaussianNB()
classifier2.fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred2)

print("Naive bayes classifier accuracy :", metrics.accuracy_score(y_test, y_pred2))

print(classification_report(y_pred2, y_test))


# Decision Tree Classifier
classifier3 = DecisionTreeClassifier()
classifier3.fit(x_train, y_train)
y_pred3 = classifier3.predict(x_test)
cm3 = confusion_matrix(y_test, y_pred3)

print("Decision tree classifier accuracy :", metrics.accuracy_score(y_test, y_pred3))

print(classification_report(y_pred3, y_test))