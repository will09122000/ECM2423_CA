from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

def main():
    # Load the digits
    digits = datasets.load_digits()

    # flatten the images
    data = digits.images.reshape((len(digits.images), -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # Randomly split data into 50% train and 50% test subsets
    xTrain, xTest, yTrain, yTest = train_test_split(
        data, digits.target, test_size=0.5)

    # Learn the digits on the train subset
    classifier.fit(xTrain, yTrain)

    # Predict the value of the digit on the test subset
    predicted = classifier.predict(xTest)

    # Display the classification report
    print("K-Means Clustering Classification Report\n\n"
          f"{metrics.classification_report(yTest, predicted)}\n")

    # Display confusion matrix with matplotlib
    confMatrix = metrics.plot_confusion_matrix(classifier, xTest, yTest)
    confMatrix.figure_.suptitle("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
