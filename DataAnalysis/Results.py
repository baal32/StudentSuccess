import pandas as pd
import matplotlib.pyplot as plt
from config import cfg

class Results(object):

    generation = None
    score_list = None


    def __init__(self, name):
        self.name = name
        self.score_list = pd.DataFrame()

    def add_result(self, score_type, score, experiment_number, target_column):
        self.score_list = self.score_list.append({'score_type': score_type, 'score': score, 'experiment': experiment_number}, ignore_index=True)

    def plot_scores(self, score_type = 'accuracy'):
        self.score_list[self.score_list['score_type'] == 'accuracy']['score'].plot()
        plt.title("Accuracy Score")
        plt.ylabel("Accuracy")
        plt.xlabel("Generation")
        plt.show()

    def plot_roc(self):
        pass

    @staticmethod
    def plot_feature_importances(feature_names, importances, indices, std):
        plt.figure()
        plt.title("Feature importances")
        plt.ylabel("Importance")
        plt.bar(range(feature_names.size), importances,
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(feature_names.size), feature_names[indices], rotation=90)
        plt.xlim([-1, feature_names.size])
        plt.tight_layout()
        # plt.show()


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