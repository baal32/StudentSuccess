from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
import logging

class Analysis(object):

    features = None
    target = None
    logger = logging.getLogger(__name__)

    def __init__(self,features, target):
        self.features = features
        self.target = target

    def random_forest(self, features, targets):
        rf = RandomForestClassifier()
        rf.fit(features, targets)

    @classmethod
    def basic_stats(self, df):
        print("Features: ", df.shape[1])
        print("Observations: ", df.shape[0])


    def logistic_regression(self, features, targets):
        pass

    def interpret_tree(self, rf, instances=None):
        if instances is None:
            instances = self.features.sample(1)
        print(rf.predict_proba(instances))
        prediction, bias, contributions = ti.predict(rf, instances)
        for i in range(len(instances)):
            print("Prediction", prediction)
            print("Bias (trainset prior)", bias)
            print("Feature contributions:")
            #
            for c, feature, target in zip(contributions[0], self.features.columns, instances.iloc[0]):
                print(feature, c, target)