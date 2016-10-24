from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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

    def fit(self, features, target):
        self.classifier.fit(features, target)



class SVMClassifier(Classifier):

    def __init__(self):
        clf = SVC()
        super().__init__(clf)

    def important_features(self,  feature_names):
        return self.clf.coef_

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
        return sorted(zip(map(lambda x: round(x, 4), self.classifier.feature_importances_), feature_names), reverse=True)