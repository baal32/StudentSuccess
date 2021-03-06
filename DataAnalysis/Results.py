import time
from platform import system

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from subprocess import check_call
from config import cfg


matplotlib.style.use('ggplot')

class Results(object):

    generation = None
    score_list = None


    def __init__(self, name):
        self.name = name
        self.score_list = pd.DataFrame()

    def add_result(self, score_type, score, experiment_number, target_column, data_type):
        self.score_list = self.score_list.append({'score_type': score_type, 'score': score, 'experiment': experiment_number, 'data_type':data_type}, ignore_index=True)
        self.target_column = target_column

    def plot_scores(self, score_type = ['Accuracy']):
        for score in score_type:
            score_df = self.score_list[self.score_list['score_type'] == score]
            x_vals = score_df['experiment'].unique()
            y_train = score_df[score_df['data_type'] == 'train']['score']
            y_test = score_df[score_df['data_type'] == 'test']['score']
            plt.figure()
            plt.title(score + " by Generation")
            plt.ylabel(score)
            plt.xlabel("Generation")
            plt.plot(x_vals, y_train)
            plt.plot(x_vals, y_test)
            plt.legend(['Train Data Scores','Test Data Scores'], loc='upper left')
            plt.savefig("Results/"+self.target_column + "_" + score + time.strftime("%d_%m_%Y_%H%M"), bbox_inches='tight')
            #plt.show()


    def plot_roc(self,clf, X_test, y_test):


        y_pred = clf.predict_proba(X_test)[:,1]
        fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred)

        roc_auc = auc(fpr_rt_lm, tpr_rt_lm)
        #print("ROC AUC %0.2f" % roc_auc)

        plt.figure()

        plt.plot(fpr_rt_lm, tpr_rt_lm, label='%s ROC curve (area = %0.2f)' % (y_test.name, roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("Results/"+self.target_column + "_ROC_" + time.strftime("%d_%m_%Y_%H%M"), bbox_inches='tight')
        #plt.show()
        '''
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Plot of a ROC curve for a specific class

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()'''

    @staticmethod
    def plot_feature_importances_bak(feature_names, importances, indices, std):
        plt.figure()
        plt.title("Feature importances")
        plt.xlabel("Importance")
        plt.barh(range(indices.size), importances[indices].reverse(),
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(indices.size), feature_names[indices].replace("_"," ").reverse(), rotation=90)
        plt.xlim([-1, indices.size])
        plt.tight_layout()
        #plt.show()

    @staticmethod
    def plot_feature_importances(feature_names, importances, indices, std, target_column):
        plt.figure(figsize=(12,6))
        plt.title("Feature importances")
        plt.xlabel("Importance")
        plt.barh(range(indices.size), importances[indices][::-1],
                color="r", yerr=std[indices][::-1], align="center")
        plt.yticks(range(indices.size), feature_names[indices]._data[::-1])
        #plt.xlim([-1, indices.size])
        plt.tight_layout()
        plt.savefig("Results/"+target_column + "_FEATURES_" + time.strftime("%d_%m_%Y_%H%M"), bbox_inches='tight')
        #plt.show()

    @classmethod
    def plot_decision_tree(cls, final_classifier,features, target_column, depth):
        dotfile = open("Results/" +target_column + "_tree.dot", 'w')
        tree.export_graphviz(final_classifier, out_file=dotfile, max_depth=depth, feature_names=features)
        dotfile.close()
        #check_call(['dot', '-Tpng', r'C:\Users\ngilbert\PycharmProjects\StudentSuccess\tree.dot', '-o',
        #            r'C:\Users\ngilbert\PycharmProjects\StudentSuccess\tree.png'])
        #check_call(['dot', '-Tpng', r'tree.dot', '-o', r'tree.png'])
        #system("dot -Tpng D:.dot -o D:/dtree2.png")

    def write_result(self, filename):
        if filename is None:
            filename = cfg['db']['results']
        #self.score_list.to_pickle("Results/"+filename)
        self.score_list.to_csv("Results/"+filename + ".csv")