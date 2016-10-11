from sklearn.ensemble import RandomForestClassifier

class Analysis(object):

    features = None
    target = None


    def __init__(self,features, target):
        self.features = features
        self.target = target

    def random_forest(self, features, targets):
        rf = RandomForestClassifier()
        rf.fit(features, targets)

    def logistic_regression(self, features, targets):
        pass