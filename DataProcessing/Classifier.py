from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

    def cross_val_score(self, features, target, scoring):
        return cross_val_score(self.classifier, features, target, scoring='accuracy')

    def fit(self, features, target):
        self.classifier.fit(features, target)

    def predict_proba(self, instances):
        return self.classifier.predict_proba(instances)



class SVMClassifier(Classifier):

    def __init__(self):
        clf = SVC(kernel='linear')
        super().__init__(clf)

    def important_features(self,  feature_names):
        return self.classifier.coef_

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