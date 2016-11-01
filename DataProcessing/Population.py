import numpy as np
import pandas as pd
import config
from copy import deepcopy
from DataAnalysis.Analysis import Analysis

class PopulationResult(object):
    def __init__(self, trained_classifier, classifier_parameters, feature_set, num_of_features, score, child_id=None):
        self.trained_classifier= trained_classifier
        self.feature_set = feature_set
        self.score = score
        self.child_id = child_id
        self.num_of_features = num_of_features
        self.classifier_params = classifier_parameters

    def __lt__(self, other):
        return self.score < other.score

# sorting class isntances
# import operator
#sorted_x = sorted(x, key=operator.attrgetter('score'))

class Population(object):
    logger = config.logger
    # list of modelresults
    model_results = []
    # list of best scoring modelresults to keep
    global_best = []

    def __init__(self):
        self.logger.debug("Initializing Model")


    def get_random_mask(self,mask_shape, column_headers, probability=.2,):
        #self.logger.info("Getting random mask, p=%f",probability)
        mask = pd.DataFrame(np.random.choice([False,True],mask_shape,p=[1-probability, probability]), columns=column_headers)
        return mask

    def add_results(self, score, features_list, classifier,classifier_parameters, child_id):
        self.logger.debug("Adding feature set with score %f to score dataframe", score)
        self.model_results.append(PopulationResult(classifier, classifier_parameters, features_list, features_list.sum(), score, child_id))
        #result = features_list
        #result["score"] = score
        #print("Feature list",features_list)
        #print("Result",result)
        #self.model_results = self.model_results.append(pd.DataFrame(result).T)


    def reset_fitness_scores_and_features(self):
        self.model_results = []

    def append_global_best_to_models(self):
        if self.global_best:
            self.logger.debug("Adding global best - #%s to model_results prior to sorting",self.global_best[0].child_id)
            self.model_results.append(self.global_best[0])

    def sort_results(self,col="score"):
        self.logger.debug("Sorting results")
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
                self.logger.debug("Updating global best - old score %f, new score %f", self.global_best[0].score , self.model_results[0].score)
            else:
                self.logger.debug("Retaining global best - existing score %f, best new score %f", self.global_best[0].score, self.model_results[0].score)
        else:
            self.global_best.append(deepcopy(top_model_result[0]))
            self.logger.debug("Setting initial global best - score %f", self.global_best[0].score)

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
            self.logger.debug("Evolving child from parents with scores %f %f",  parent1.score, parent2.score)
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
        #self.logger.debug("Mutation mask with mutating probability %f, mask = %s", mutation_probability, mutation_mask)
        mutated_child = np.logical_xor(mutation_mask, child)
        #self.logger.info("Premutation child features: %d, mutation mask features: %d, mutated child features: %d", (child > 0).sum(), np.sum(mutation_mask), (mutated_child > 0).sum())
        #self.logger.info("Mutated child %s", mutated_child)
        return mutated_child

    def crossover(self, parent1_features, parent2_features):
        crossover_point = np.random.randint(0,parent1_features.shape[0])
        crossover = pd.concat([parent1_features[ 0:crossover_point], parent2_features[ crossover_point:]], axis=0)
        return crossover

    def purge_low_scores(self, population_purge_pct):
        #self.logger.info("Purging %f population before purge count: %d", population_purge_pct, len(self.model_results))
        del self.model_results[int(len(self.model_results)*(1- population_purge_pct)):]
        #self.model_results = self.model_results[:int((self.model_results.shape[0] * (1 - population_purge_pct)))]
        self.logger.debug("Purging - population after purge count: %d", len(self.model_results))

    def print_best_results(self, retain_best, a):
        best_results = self.get_best_feature_sets(retain_best)
        for i in best_results:
            self.logger.info("Child: %s Final score: %f Features: %s", i.child_id, i.score, i.trained_classifier.important_features(i.feature_set[i.feature_set].index))
#                        a.important_features(i.trained_classifier, i.feature_set[i.feature_set].index))

#        self.high_scores.append(pd.DataFrame({score: features_list}))




