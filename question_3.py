import pydotplus
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image

def main():
    featureColumns = ["Roads:number_intersections",
                    "Roads:diversity",
                    "Roads:total",
                    "Buildings:diversity",
                    "Buildings:total",
                    "LandUse:Mix",
                    "TrafficPoints:crossing",
                    "poisAreas:area_park",
                    "poisAreas:area_pitch",
                    "pois:diversity",
                    "pois:total",
                    "ThirdPlaces:oa_count",
                    "ThirdPlaces:edt_count",
                    "ThirdPlaces:out_count",
                    "ThirdPlaces:cv_count",
                    "ThirdPlaces:diversity",
                    "ThirdPlaces:total",
                    "vertical_density",
                    "buildings_age",
                    "buildings_age:diversity"]

    classNames = ["age_under_18",
                "age_18_30",
                "age_30_40",
                "age_40_50",
                "age_50_60",
                "age_over_60"]

    # Split the data in a training and testing set
    x = pd.read_csv("./data.csv", usecols=featureColumns)
    y = pd.read_csv("./data.csv", usecols=["most_present_age"])
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3,
                                                        random_state = 0)

    # Create the decision tree using entropy
    dtc = DecisionTreeClassifier(criterion = "entropy")
    clf = dtc.fit(X_train,Y_train)
    clf.score(X_test, Y_test)

    # Calculate the accuracy for varying depth of the tree and see how it
    # changes
    accuracyArray = []
    for depth in range(1, 100):
        dtc = DecisionTreeClassifier(max_depth = depth)
        dtc = dtc.fit(X_train, Y_train)
        accuracy = metrics.accuracy_score(Y_test, dtc.predict(X_test))
        accuracyArray.append(accuracy)

    # Plot decision tree accuracy graph
    accuracyGraph = plt.figure()
    plt.plot(range(1, 100), accuracyArray)
    plt.title("Classifier Accuracy against Depth of Decision Tree")
    plt.xlabel("Decision Tree Depth")
    plt.ylabel("Classifier Accuracy")

    # Display the most important features for a fitted tree ranked
    importances = []
    names = []
    for importance, name in sorted(zip(clf.feature_importances_, X_train.columns),
                                   reverse=True):
        if (importance > 0):
            importances.append(importance)
            names.append(name)

    # Display bar graph of features
    FeaturesGraph = plt.figure()
    plt.bar(names, importances)
    plt.title("Most Important Features that determine Age Range")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=90)

    # Create decision tree image
    data = StringIO()
    export_graphviz(clf, out_file=data, filled=True, rounded=True,
                    special_characters=True, feature_names = featureColumns,
                    class_names = classNames)
    graph = pydotplus.graph_from_dot_data(data.getvalue())
    graph.write_png("Decision Tree.png")
    Image(graph.create_png())

    plt.show()

if __name__ == "__main__":
    main()
