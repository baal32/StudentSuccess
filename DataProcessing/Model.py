import numpy as np
import pandas as pd

class Model(object):


    high_scores = pd.DataFrame()
    global_best = pd.DataFrame()

    def __init__(self):
        pass



    @staticmethod
    def get_random_mask(mask_length, probability=.2):
        return pd.DataFrame(np.random.choice([False,True],mask_length,p=[1-probability, probability]))[0]

    def add_results(self, score, features_list):
        result = features_list
        result["score"] = score
        print("Feature list",features_list)
        print("Result",result)
        self.high_scores = self.high_scores.append(pd.DataFrame(result).T)


    def sort_results(self,col="score)"):
        self.high_scores.sort_values(col, ascending=False)

    def get_best_features_sets(self, num_of_best=2):
        sorted_scores = self.high_scores.sort_values(['score'], ascending=False)
        print(sorted_scores)
        return sorted_scores[0:num_of_best]
        #return parents[:1],parents[1:2]

    def evaluate_global_best(self):
        self.global_best = self.global_best.append(pd.DataFrame(result).T)

    def evolve_children(self, parents, num_of_children):
        children = self.crossover(parents, num_of_children)
        self.mutate(children, .015)

    def mutate(self, children, mutation_probability=0.05):
        mutation_mask = np.random.choice([True,False],children.shape[1],p=[mutation_probability, 1-mutation_probability])


        pass

    def crossover(self, parents, num_of_children):
        pass

#        self.high_scores.append(pd.DataFrame({score: features_list}))




