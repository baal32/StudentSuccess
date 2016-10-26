from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

"""
    def factory(type):
        if type == 'SVM': return SVMClassifier()
        if type == 'RandomForestClassifier': return RFClassifier()

    factory = staticmethod(factory)
"""


class Classifier(object):

    classifier = None

    def __init__(self, classifier):
        self.classifier = classifier

    def get_params(self):
        return self.classifier.get_params()

    def score(self, features, target):
        return self.classifier.score(features, target)

    def cross_val_score(self, features, target, scoring='accuracy'):
        return cross_val_score(self.classifier, features, target, scoring)

    def fit(self, features, target):
        self.classifier.fit(features, target)

    def predict_proba(self, instances):
        return self.classifier.predict_proba()



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
            self.logger.info("Child: %s Final score: %f Features: %s", i.child_id, i.score,
                             a.important_features(i.trained_classifier, i.feature_set[i.feature_set].index))

class RFClassifier(Classifier):
    def __init__(self):
        clf = RandomForestClassifier()
        super().__init__(clf)

    def print_features(self):
        print ("features of rf")

    def important_features(self, feature_names):
        return sorted(zip(map(lambda x: "%.3f" % round(x, 4), self.classifier.feature_importances_), feature_names), reverse=True)

class KNClassifier(Classifier):
    def __init__(self):
        clf = KNeighborsClassifier()
        super().__init__(clf)

    def important_features(self, feature_names):
        print("important features")