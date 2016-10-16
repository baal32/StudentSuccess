from DataCollection.DataSource import DataSource
from DataProcessing.Preprocessor import Processor
from DataAnalysis.Plotter import Plotter
from DataAnalysis.Analysis import Analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging
from DataProcessing.Model import Model
import cProfile
import csv
import config




def main():
    """ Run the main program"""
    logger = logging.getLogger(__name__)
    #extract data (instantiate and call method at same time)
    full_frame = DataSource().get_all_data()

    Analysis.basic_stats(full_frame)
    #preprocess data
    preprocessor = Processor(full_frame)
    full_frame = preprocessor.one_hot()
    full_frame = preprocessor.drop_columns()
    full_frame = preprocessor.impute_missing_values()
    full_frame = preprocessor.drop_rows_with_NA()



    #split into train and test


    #print(train_features.columns[0:7])

    # Model instnace
    model = Model()

    # genetic feature selection and scoring loop

    # set number of iterations


    train_features, train_target, test_features, test_target = preprocessor.split_test_train_features_targets(.75,  specific_target="APROG_PROG_STATUS")

    experiments = 10
    population_size = 10
    retain_best = 3
    low_score_purge_pct = .5
    initial_population_chance_bit_on = .05
    #initial population randomly generated


    # experiments
    for experiment_number in range(experiments):

        config.logger.info("Experiment #%d", experiment_number+1)
        # create the population that will be analyzed
        if experiment_number == 0:
            population = model.get_random_mask((population_size, train_features.shape[1]), probability=initial_population_chance_bit_on, column_headers = train_features.columns.values )
            #population = train_features[train_features.columns[feature_mask]]



        # evaluate population
        # for i in range(population_size):

        # for each experiment reset the score dataframe to be population by the newest population
        model.reset_fitness_scores_and_features()
        for i,p  in population.iterrows():
            analysis = Analysis(test_features, test_target)

            #print("Train features ",train_features)
            #print("Train feature subsetted ",population)
            train_features_subset = train_features[train_features.columns[p]]
            test_features_subset = test_features[test_features.columns[p]]
            #print(population)
            #print(test_features_subset.shape[1], " Features chosen ", population.columns.values)

            clf = RandomForestClassifier(n_jobs=10, n_estimators=100) #, max_depth=10)
            clf = clf.fit(train_features_subset, train_target)
            score = clf.score(test_features_subset, test_target)
            print("# Features chosen: ", test_features_subset.shape[1], " score: ", score, "top 3 features: ", analysis.important_features(clf, population)[0:3])
            # save score and features to experiment set
            #print("Feature mask",feature_mask)
            model.add_results(score, p)
            # get feature set


        # sort scores
        model.sort_results()
        # throw away lowest scorers (population_purge) by pct
        model.purge_low_scores(population_purge_pct=low_score_purge_pct)
        # take the rest of the scores, append the global best, and determine new global best
        model.evaluate_global_best()
        # evolve children from remaining population
        population = model.evolve_children(population_size)
        logger.info("Global best by experiment %d: %f", experiment_number, model.global_best.iloc[0]['score'])
        # now that scores have been evaluated, purge poor scorers

        # determine best 2 results from experiment set
        #parents = model.get_best_feature_sets(retain_best)
        #for i,p in parents.iterrows():
            #print("P",p)
            #print("Parent:",i,p)
            #print(type(p))
        #    print("Parent ",i,"score:",p['score']," features:",train_features.columns[p.drop('score') == 1.0].values)
#            print(parents[p:p+1])
#        print("Parent1",parents[:1], "Parent2",parents[1:2])

    best_results = model.get_best_feature_sets(retain_best)
    for i,p in best_results.iterrows():
        print("Final best ",i,"score ",p['score'],"***************************************************\n\n\n\n")
        #use crossover and mutation to get children
   #     child1, child2 = model.evolve_children(parents,2)

        #compare to global best and determine new global best








    #Plotter.histogram(clf,train_features)
    print(clf)

    print(clf.score(test_features,test_target))

    analysis = Analysis(test_features, test_target)
    print(analysis.important_features(clf,train_features))
    analysis.interpret_tree(clf)
  #  loc_submission = "test.csv"
  #  with open(loc_submission, "w") as outfile:
  #      writer = csv.writer(outfile)
  #      writer.writerow(["Id,Outcome"])
  #      for e, val in enumerate(list(clf.predict(test_features))):
  #          writer.writerow([test_emplids[e],val])
    #genetic feature subselection
    #create random forest and fit
    #repeat using feature subsetting

    #interpet tree


    #interpret other metrics

    #find most important metrics

    #display or visualize
    """
    extract_data()
    cache_data()

    run_decision_tree()
    analyze_results()
    pass
    """



cProfile.run(main())