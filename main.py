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

    experiments = 2
    population_size = 4
    retain_best = 3

    # experiments
    for i in range(experiments):

        # evaluate population
        for i in range(population_size):
            feature_mask = Model.get_random_mask(train_features.shape[1], probability=.1)
            train_features_subset = train_features[train_features.columns[feature_mask]]
            #print("Train features ",train_features)
            #print("Train feature subsetted ",train_features_subset)
            test_features_subset = test_features[test_features.columns[feature_mask]]
            #print(train_features_subset)
            print(test_features_subset.shape[1]," Features chosen ", train_features_subset.columns.values)
            clf = RandomForestClassifier(n_jobs=10, n_estimators=100) #, max_depth=10)
            clf = clf.fit(train_features_subset, train_target)
            score = clf.score(test_features_subset, test_target)

            # save score and features to experiment set
            #print("Feature mask",feature_mask)
            model.add_results(score, feature_mask)

            print(score)
            analysis = Analysis(test_features, test_target)
            print(analysis.important_features(clf, train_features_subset))
            # get feature set


        # determine best 2 results from experiment set
        parents = model.get_best_features_sets(retain_best)
        for i,p in parents.iterrows():
            print("P",p)
            print("Parent:",i,p)
            print("Parent ",i," features:",train_features.columns[p].values)
#            print(parents[p:p+1])
#        print("Parent1",parents[:1], "Parent2",parents[1:2])

        #use crossover and mutation to get children
        child1, child2 = model.evolve_children(parents,2)

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