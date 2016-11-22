import numpy as np
import matplotlib.pyplot as plt


class Plotter(object):

    @classmethod
    def histogram(cls,classifier, features):
        importances = classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]

        indices = indices[0:7]
        print(indices)
        print(features.columns[0:7])
        print(features.columns[indices])
        # Print the feature ranking
        print("Feature ranking:")

#        for f in range(features.shape[1]):
        for f in range(len(indices)):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(indices)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), features.columns[indices])
        plt.xlim([-1,len(indices)])
        plt.show()