from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
import numpy as np



import config

class Analysis(object):

    features = None
    target = None
    logger =config.logger

    def __init__(self,features, target):
        self.features = features
        self.target = target

    @classmethod
    def basic_stats(self, df):
        self.logger.info("Dataset shape: %s, features: %d, observations: %d", df.shape, df.shape[1],df.shape[0])
        self.logger.info("Completions: %d, Discontinuations: %d, Completion pct: %f", len(df[df["APROG_PROG_STATUS"] == 'CM']), len(df[df["APROG_PROG_STATUS"] == 'DC']), len(df[df["APROG_PROG_STATUS"] == 'CM'])/float(df.shape[0]))

    def logistic_regression(self, features, targets):
        pass

    def important_features(self,clf, feature_names):
        return sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True)

    def interpret_tree(self, rf, instances=None):
        if instances is None:
            instances = self.features.sample(1)
        #print(rf.predict_proba(instances))
        prediction, bias, contributions = ti.predict(rf, instances)
        for i in range(len(instances)):
            self.logger.info("Prediction %s ----------------------------------------", prediction[i])
            self.logger.info("Bias (trainset prior) %s", bias[i])
            self.logger.info("Feature contributions:")
            #for c, feature, actual in sorted(
             #       zip(contributions[i], instances.columns, instances.iloc[i]),
             #       key=lambda x: -abs(x[0])):
             #   self.logger.info("%s Contribution: %f Actual: %f",feature, round(c, 3), round(actual,3))
            for c, feature, actual in sorted(  zip(contributions[i], instances.columns, instances.iloc[i]), key=lambda x: -abs(x[0][0]))[0:5]:
                # TODO fix rounding issue
                # self.logger.info("%s Contribution: %s Actual: %f", feature, ["%.3f" % a for a in c], round(actual, 3))
                self.logger.info("%s Contribution: %s Actual: %s", feature, c, actual)
            #feature_strength = zip(contributions[i], instances.columns, instances.iloc[i])
            #print(sorted(feature_strength, key=lambda x: x[0].max))
#            for c, feature, target in zip(contributions[i], instances.columns, instances.iloc[i]):
#                print(sorted(feature, c, target)