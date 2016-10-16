import numpy as np
import pandas as pd
import config

class Model(object):


    logger = config.logger
    fitness_scores_and_features = pd.DataFrame()
    global_best = pd.DataFrame({"score": 0.0}, index=[0])

    def __init__(self):
        self.logger.info("Initializing Model")


    def get_random_mask(self,mask_shape, probability=.2):
        #self.logger.info("Getting random mask, p=%f",probability)
        mask = pd.DataFrame(np.random.choice([False,True],mask_shape,p=[1-probability, probability]))
        return mask

    def add_results(self, score, features_list):
        self.logger.info("Adding feature with score %f to score dataframe", score)
        result = features_list
        result["score"] = score
        #print("Feature list",features_list)
        #print("Result",result)
        self.fitness_scores_and_features = self.fitness_scores_and_features.append(pd.DataFrame(result).T)


    def sort_results(self,col="score"):
        if self.global_best is not None:
            self.fitness_scores_and_features.append(self.global_best)
        self.fitness_scores_and_features.sort_values(col, ascending=False, inplace=True)

    def get_best_feature_sets(self, num_of_best=2):
        sorted_scores = self.fitness_scores_and_features.sort_values(['score'], ascending=False)
        #print(sorted_scores)
        #self.logger.info("Getting best features - score %f - features %s",sorted_scores.iloc[0:num_of_best]['score'],'TODO: add feature list')
        return sorted_scores[0:num_of_best]
        #return parents[:1],parents[1:2]

    def evaluate_global_best(self):
        if (self.global_best is None) or ((self.global_best[:1]['score'] < self.fitness_scores_and_features[0:1]['score'])[0]):
            self.logger.info("Updating global best - old score %f, new score %f",self.global_best[:1]['score'], self.fitness_scores_and_features[0:1]['score'])
            self.global_best.iloc[0] = self.fitness_scores_and_features.iloc[0]
        #self.global_best = self.global_best.append(pd.DataFrame(result).T)

    def evolve_children(self, num_of_children):
        # for i in range num_of_children, evolve a child and add to the dataframe
        children_df = pd.DataFrame()

        for i in range(num_of_children):
            # get first parent
            parent1 = self.fitness_scores_and_features.iloc[np.random.randint(0, self.fitness_scores_and_features.shape[0], size=1)]
            self.logger.info("Chose parent 1 with score %f", parent1['score'])
            # get second parent
            parent2 = self.fitness_scores_and_features.iloc[np.random.randint(0, self.fitness_scores_and_features.shape[0], size=1)]
            while parent1.equals(parent2):
                parent2 = self.fitness_scores_and_features.iloc[np.random.randint(0, self.fitness_scores_and_features.shape[0], size=1)]

            self.logger.info("Chose parent 2 with score %f", parent2['score'])

            # perform crossover
            self.logger.info("Evolving child from parents with scores %f %f",  parent1['score'], parent2['score'])
            new_child = self.crossover(parent1, parent2)
            self.mutate(new_child, .15)
            # add the resultant child to the dataframe
            self.logger.info("Adding child %d",i)
            children_df.append(new_child)
        return children_df

    def evolve_child(self, parent1, parent2):
        pass

    def mutate(self, children, mutation_probability=0.15):
        mutation_mask = np.random.choice([True,False],children.shape[1],p=[mutation_probability, 1-mutation_probability])
        #self.logger.info("Mutation mask with mutating probability %f, mask = %s", mutation_probability, mutation_mask)
        mutated_child = np.logical_xor(mutation_mask, children)
        #self.logger.info("Mutated child %s", mutated_child)
        return mutated_child

    def crossover(self, parent1, parent2):
        # one point crossover
        crossover_point = np.random.randint(0,parent1.shape[1])
        #print("Parent1:",parent1.iloc[0],"Parent2:",parent2.iloc[0])
        return pd.concat([parent1.ix[:,0:crossover_point],parent2.ix[:,crossover_point:]], axis=1)

    def purge_low_scores(self, population_purge_pct):
        self.logger.info("Purging %f population before purge count: %d", population_purge_pct, self.fitness_scores_and_features.shape[0])
        self.fitness_scores_and_features = self.fitness_scores_and_features[:int((self.fitness_scores_and_features.shape[0] * (1-population_purge_pct)))]
        self.logger.info("Purging - population after purge count: %d", self.fitness_scores_and_features.shape[0])

#        self.high_scores.append(pd.DataFrame({score: features_list}))




