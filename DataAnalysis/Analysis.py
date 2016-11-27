import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from treeinterpreter import treeinterpreter as ti
from DataAnalysis.Results import Results
import numpy as np
import pandas as pd


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

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def classification_scores(y_test, y_predict, target_column):
        accuracy = ["Accuracy", accuracy_score(y_test[target_column], y_predict)]
        f1 = ["F1 Score", f1_score(y_test[target_column], y_predict, pos_label=1)]
        precision = ["Precision", precision_score(y_test[target_column], y_predict, pos_label=1)]
        recall = ["Recall", recall_score(y_test[target_column], y_predict, pos_label=1)]
        matthews = ["Matthews Coefficient", matthews_corrcoef(y_test[target_column], y_predict)]
        confusion = ["Confusion Matrix", confusion_matrix(y_test[target_column], y_predict)]
        roc = ["ROC Score", roc_auc_score(y_test[target_column], y_predict)]
        scores = [accuracy, f1, precision, recall, matthews, roc]
        # logger.info("%s", scores)
        # logger.info("%s", classification_report)
        return scores
        # for score in scores:
        #    logger.info("%s %s", score[0], score[1])

    @staticmethod
    def column_correlation(feature_columns, target_column):
        lb = LabelEncoder()
        target_column = pd.Series(lb.fit_transform(target_column))
        df2 = pd.DataFrame(columns=["Column", "Correlation"])
        for column in feature_columns:
            try:
                df2.loc[df2.shape[0]] = [column, np.abs(feature_columns[column].corr(target_column))]
                # print(column, df[column].corr(df["APROG_PROG_STATUS"]))
            except:
                pass
        return df2.sort_values(by=['Correlation'],ascending=False)

    @staticmethod
    def nulls_by_feature(features):
        print(features.isnull().sum().sort_values(ascending=False))

    @staticmethod
    def agg_by_target(feature_column, target_column,aggregation_method = 'AVG'):
        df = pd.DataFrame({feature_column.name: feature_column, target_column.name: target_column},dtype=float )
        config.logger.debug("%s %s %s %s", aggregation_method, feature_column.index, target_column.index, df.groupby(target_column.name)[feature_column.name].mean())

    @classmethod
    def important_features(cls,tree_classifier, feature_names, target_column, threshold=0.05):
        importances = tree_classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in tree_classifier.estimators_], axis=0)
        #        mean = np.mean([tree.tree_.threshold for tree in self.classifier.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        significant_feature_index = indices[np.where(importances[indices] > threshold)]
        # Print the feature ranking
        cls.logger.debug("Feature ranking:")

        for f in range(significant_feature_index.size):
            cls.logger.debug("%d. feature %d %s (%f)", f + 1, significant_feature_index[f],
                        feature_names[significant_feature_index[f]], importances[significant_feature_index[f]])

        # Plot the feature importances of the forest
        Results.plot_feature_importances(feature_names,  importances, significant_feature_index, std, target_column)

        return sorted(zip(map(lambda x: "%.3f" % round(x, 4), tree_classifier.feature_importances_), feature_names),
                     reverse=True)

    #def important_features(self,clf, feature_names):
    #    return sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names), reverse=True)

    def interpret_tree(self, rf, instances, full_frame):
        identifiers = full_frame.loc[instances.index,["EMPLID", "GRADUATED"]]
        #with open('prediction_instances.csv','wb') as csvfile:


        if instances is None:
            instances = self.features.sample(1)
        #print(rf.predict_proba(instances))
        prediction, bias, contributions = ti.predict(rf, instances)
        #prediction, bias, contributions = sorted(zip(ti.predict(rf, instances)), key=lambda x: -x[0][0])

        #df = pd.DataFrame({"Prediction":prediction, "Bias":bias, "Contribution":contributions})
        #for r in df[df['Prediction'][0] > 0.5].sort_values(by="Prediction", ascendingsorted(df[df['Prediction'][0] > 0.5], )
#        temp = instances
#        temp["EMPLID"] = emplids

        predicted_class = pd.DataFrame(np.apply_along_axis((lambda x: x[1] > x[0]),axis=1, arr=prediction).astype(int), index=instances.index)
        instances_with_results = pd.concat([identifiers, predicted_class, instances], axis=1)
        instances_with_results.to_csv('prediction_instances.csv')
        #sorted_list = list(sorted(zip(prediction, bias, contributions), key=lambda x: -x[0][0]))
        #for prediction, bias, contributions in sorted(zip(prediction, bias, contributions), key=lambda x: -x[0][0]):
        prediction_index = 0
        for i in range(len(instances)): #instances_with_results.index: #
            if np.abs(prediction[i][1] - prediction[i][0]) <= 0.8:
                continue
            self.logger.info("Prediction for %s - %s ----------------------------------------", instances_with_results.iloc[i]["EMPLID"],prediction[i])
            self.logger.info("Bias (trainset prior) %s", bias[i])
            self.logger.info("Feature contributions:")
            #for c, feature, actual in sorted(
             #       zip(contributions[i], instances.columns, instances.iloc[i]),
             #       key=lambda x: -abs(x[0])):
             #   self.logger.info("%s Contribution: %f Actual: %f",feature, round(c, 3), round(actual,3))
            for c, feature, actual in sorted(  zip(contributions[i], instances.columns, instances.iloc[i]), key=lambda x: -abs(x[0][0]))[0:10]:
                # TODO fix rounding issue
                # self.logger.info("%s Contribution: %s Actual: %f", feature, ["%.3f" % a for a in c], round(actual, 3))
                self.logger.info("%s Contribution: %s Actual: %s", feature, c, actual)
            prediction_index = prediction_index + 1
            #feature_strength = zip(contributions[i], instances.columns, instances.iloc[i])
            #print(sorted(feature_strength, key=lambda x: x[0].max))
#            for c, feature, target in zip(contributions[i], instances.columns, instances.iloc[i]):
#                print(sorted(feature, c, target)

    @staticmethod
    def crosstab(y_actual, y_predict):
        print(pd.crosstab(y_actual, y_predict, rownames=['actual'], colnames=['preds']))

