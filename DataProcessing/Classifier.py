from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
import matplotlib.pyplot as plt
import config
from DataAnalysis.Results import Results

"""
    def factory(type):
        if type == 'SVM': return SVMClassifier()
        if type == 'RandomForestClassifier': return RFClassifier()

    factory = staticmethod(factory)
"""
logger = config.logger

class Classifier(object):

    classifier = None

    def __init__(self, classifier):
        self.classifier = classifier

    def get_params(self):
        return self.classifier.get_params()

    def set_params(self, **kwargs):
        return self.classifier.set_params(**kwargs)

    def score(self, features, target):
        return self.classifier.score(features, target)

    def cross_val_score(self, features, target, scoring='accuracy'):
        return cross_val_score(self.classifier, features, target, scoring)

    def fit(self, features, target):
        self.classifier.fit(features, target)

    def predict_proba(self, instances):
        return self.classifier.predict_proba(instances)


    def predict(self, instances):
        return self.classifier.predict(instances)


class SVMClassifier(Classifier):

    def __init__(self):
        n_estimators = 10
        clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
        super().__init__(clf)

    def important_features(self,  feature_names):
        pass #TODO return self.classifier.coef_

    def print_best_results(self, retain_best, a):
        best_results = self.get_best_feature_sets(retain_best)
        for i in best_results:
            logger.info("Child: %s Final score: %f Features: %s", i.child_id, i.score,
                             a.important_features(i.trained_classifier, i.feature_set[i.feature_set].index))

class LinearSVCClassifier(Classifier):

    def __init__(self):
        n_estimators = 10
        clf = LinearSVC()
        super().__init__(clf)

    def important_features(self,  feature_names):
        pass #TODO return self.classifier.coef_

    def print_best_results(self, retain_best, a):
        best_results = self.get_best_feature_sets(retain_best)
        for i in best_results:
            logger.info("Child: %s Final score: %f Features: %s", i.child_id, i.score,
                             a.important_features(i.trained_classifier, i.feature_set[i.feature_set].index))

class RFClassifier(Classifier):
    def __init__(self):
        clf = RandomForestClassifier(n_jobs=8)
        super().__init__(clf)

    def print_features(self):
        print ("features of rf")

    def important_features(self, feature_names, threshold=0.01):
        importances = self.classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)
        #        mean = np.mean([tree.tree_.threshold for tree in self.classifier.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        significant_feature_index = indices[np.where(importances[indices] > threshold)]
        # Print the feature ranking
        logger.info("Feature ranking:")

        for f in range(significant_feature_index.size):
            logger.info("%d. feature %d %s (%f)", f + 1, significant_feature_index[f],
                        feature_names[significant_feature_index[f]], importances[significant_feature_index[f]])

        # Plot the feature importances of the forest
        Results.plot_feature_importances(feature_names,  importances, significant_feature_index, std)

        return sorted(zip(map(lambda x: "%.3f" % round(x, 4), self.classifier.feature_importances_), feature_names),
                     reverse=True)


   #def important_features(self, feature_names):
   #    return sorted(zip(map(lambda x: "%.3f" % round(x, 4), self.classifier.feature_importances_), feature_names), reverse=True)

class ETClassifier(Classifier):
   def __init__(self):
       clf = ExtraTreesClassifier()
       super().__init__(clf)

   def print_features(self):
       print("features of ef")

   def important_features(self, feature_names, threshold = 0.01):
       importances = self.classifier.feature_importances_
       std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)
#        mean = np.mean([tree.tree_.threshold for tree in self.classifier.estimators_], axis=0)
       indices = np.argsort(importances)[::-1]

       significant_feature_index = indices[np.where(importances[indices] > threshold)]
       # Print the feature ranking
       logger.info("Feature ranking:")

       for f in range(significant_feature_index.size):
           logger.info("%d. feature %d %s (%f)", f + 1, significant_feature_index[f], feature_names[significant_feature_index[f]], importances[significant_feature_index[f]])

       # Plot the feature importances of the forest
       '''
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(feature_names.size), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(feature_names.size), feature_names[indices], rotation=90)
        plt.xlim([-1,feature_names.size])
        plt.tight_layout()
        #plt.show()
        #return sorted(zip(map(lambda x: "%.3f" % round(x, 4), self.classifier.feature_importances_), feature_names),
        #              reverse=True)
        '''


class KNClassifier(Classifier):
    def __init__(self):
        clf = KNeighborsClassifier()
        super().__init__(clf)

    def important_features(self, feature_names):
        print("important features")