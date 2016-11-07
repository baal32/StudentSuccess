import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
from config import cfg

matplotlib.style.use('ggplot')

class Results(object):

    generation = None
    score_list = None


    def __init__(self, name):
        self.name = name
        self.score_list = pd.DataFrame()

    def add_result(self, score_type, score, experiment_number, target_column):
        self.score_list = self.score_list.append({'score_type': score_type, 'score': score, 'experiment': experiment_number}, ignore_index=True)

    def plot_scores(self, score_type = ['Accuracy']):
        for score in score_type:
            score_df = self.score_list[self.score_list['score_type'] == score]
            x_vals = score_df['experiment']
            y_vals = score_df['score']
            plt.title(score + " by Generation")
            plt.ylabel(score)
            plt.xlabel("Generation")
            plt.plot(x_vals, y_vals)
            plt.show()


    @staticmethod
    def plot_roc(clf, X_test, y_test):


        y_pred = clf.predict_proba(X_test)[:,1]
        fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred)

        roc_auc = auc(fpr_rt_lm, tpr_rt_lm)
        print("ROC AUC %0.2f" % roc_auc)

        plt.figure()

        plt.plot(fpr_rt_lm, tpr_rt_lm, label='%s ROC curve (area = %0.2f)' % (y_test.name, roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
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
    def plot_feature_importances(feature_names, importances, indices, std):
        plt.figure()
        plt.title("Feature importances")
        plt.ylabel("Importance")
        plt.bar(range(indices.size), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(indices.size), feature_names[indices], rotation=90)
        plt.xlim([-1, indices.size])
        plt.tight_layout()
        #plt.show()


    def write_result(self, filename):
        if filename is None:
            filename = cfg['db']['results']
        self.score_list.to_pickle(filename)

    def plot_results(self):
        plt.figure().set_size_inches(8, 6)
        plt.semilogx(alphas, scores)

        # plot error lines showing +/- std. errors of the scores
        std_error = scores_std / np.sqrt(n_folds)

        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])