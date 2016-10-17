import numpy as np
import pandas as pd
import config
from copy import deepcopy

class ModelResult(object):
    def __init__(self, trained_classifier, feature_set, score):
        self.trained_classifier= trained_classifier
        self.feature_set = feature_set
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

# sorting class isntances
# import operator
#sorted_x = sorted(x, key=operator.attrgetter('score'))

class Model(object):


    logger = config.logger

    # list of modelresults
    model_results = []
    #global_best = pd.DataFrame({"score": 0.0}, index=[0])

    # list of best scoring modelresults to keep
    global_best = []

    def __init__(self):
        self.logger.info("Initializing Model")


    def get_random_mask(self,mask_shape, column_headers, probability=.2,):
        #self.logger.info("Getting random mask, p=%f",probability)
        mask = pd.DataFrame(np.random.choice([False,True],mask_shape,p=[1-probability, probability]), columns=column_headers)
        return mask

    def add_results(self, score, features_list, classifier):
        #self.logger.info("Adding feature set with score %f to score dataframe", score)
        self.model_results.append(ModelResult(classifier, features_list,score))
        #result = features_list
        #result["score"] = score
        #print("Feature list",features_list)
        #print("Result",result)
        #self.model_results = self.model_results.append(pd.DataFrame(result).T)


    def reset_fitness_scores_and_features(self):
        self.model_results = []


    def sort_results(self,col="score"):
        self.logger.info("Sorting results")
        if self.global_best:
            self.model_results.append(self.global_best[0])
        self.model_results.sort(reverse=True)

    def get_best_feature_sets(self, num_of_best=1):
        self.sort_results()
        return self.model_results[:num_of_best]
        #sorted_scores = self.model_results.sort_values(['score'], ascending=False)
        #print(sorted_scores)
        #self.logger.info("Getting best features - score %f - features %s",sorted_scores.iloc[0:num_of_best]['score'],'TODO: add feature list')
        #return sorted_scores[0:num_of_best]
        #return parents[:1],parents[1:2]

    def evaluate_global_best(self):
        top_model_result = self.get_best_feature_sets(num_of_best = 1)
        if self.global_best:
            if self.global_best[0] < top_model_result[0]:
                self.global_best[0] = deepcopy(top_model_result[0])
                self.logger.info("Updating global best - old score %f, new score %f", self.global_best[0].score , self.model_results[0].score)
            else:
                self.logger.info("Retaining global best - existing score %f, best new score %f", self.global_best[0].score, self.model_results[0].score)
        else:
            self.global_best.append(deepcopy(top_model_result[0]))
            self.logger.info("Setting initial global best - score %f", self.global_best[0].score)

    def evolve_children(self, num_of_children):
        # for i in range num_of_children, evolve a child and add to the dataframe
        children_df = pd.DataFrame()

        for i in range(num_of_children):
            # get first parent
            parent1 = self.model_results[np.random.randint(0, len(self.model_results))]
            #self.logger.info("Chose parent 1 with score %f", parent1['score'])
            # get second parent
            parent2 = self.model_results[np.random.randint(0, len(self.model_results))]
            while parent1.feature_set.equals(parent2.feature_set):
                #self.logger.info("Selected same result as both parents, reselecting 2nd parent")
                parent2 = self.model_results[np.random.randint(0, len(self.model_results))]

            #self.logger.info("Chose parent 2 with score %f", parent2['score'])

            # perform crossover
            self.logger.info("Evolving child from parents with scores %f %f",  parent1.score, parent2.score)
            new_child = self.crossover(parent1.feature_set, parent2.feature_set)
            new_child = self.mutate(new_child, config.cfg['genetic']['mutation_rate'])
            #drop score column so remaining should just be 1s and 0s
            #new_child.drop(['score'], axis=1, inplace=True)

            #convert to boolean
            new_child = (new_child == 1)

            # add the resultant child to the dataframe
            #self.logger.info("Adding child %d",i)
            children_df = children_df.append([new_child], i)
        return children_df

    def evolve_child(self, parent1, parent2):
        pass

    def mutate(self, child, mutation_probability=0.02):
        mutation_mask = np.random.choice([True,False], child.shape[0], p=[mutation_probability, 1 - mutation_probability])
        #self.logger.info("Mutation mask with mutating probability %f, mask = %s", mutation_probability, mutation_mask)
        mutated_child = np.logical_xor(mutation_mask, child)
        self.logger.info("Premutation child features: %d, mutation mask features: %d, mutated child features: %d", (child > 0).sum(), np.sum(mutation_mask), (mutated_child > 0).sum())
        #self.logger.info("Mutated child %s", mutated_child)
        return mutated_child

    def crossover(self, parent1_features, parent2_features):
        # one point crossover
        crossover_point = np.random.randint(0,parent1_features.shape[0])
        #print("Parent1:",parent1.iloc[0],"Parent2:",parent2.iloc[0])
        # need to reindex both parents otherwise when concatting you end up with two rows, one of which has the values for parent1 and other cols NaN and the other which has the values for parent2 and teh rest NaN
        # index      col1      col2    (crossover point)  col3      col4
        #   3         1         0                          NA         NA
        #   5         NA        NA                         0          1
        #parent1_features.index = [0]
        #parent2_features.index = [0]
        #crossover = pd.concat([parent1_features.ix[:, 0:crossover_point], parent2_features.ix[:, crossover_point:]], ignore_index=True, axis=1)
        crossover = pd.concat([parent1_features[ 0:crossover_point], parent2_features[ crossover_point:]], axis=0)
        return crossover

    def purge_low_scores(self, population_purge_pct):
        self.logger.info("Purging %f population before purge count: %d", population_purge_pct, len(self.model_results))
        del self.model_results[int(len(self.model_results)*(1- population_purge_pct)):]
        #self.model_results = self.model_results[:int((self.model_results.shape[0] * (1 - population_purge_pct)))]
        self.logger.info("Purging - population after purge count: %d", len(self.model_results))

#        self.high_scores.append(pd.DataFrame({score: features_list}))




