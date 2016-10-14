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


    # genetic feature selection and scoring loop

    # set number of iterations
    iterations = 10
    train_features, train_target, test_features, test_target = preprocessor.split_test_train_features_targets(.75, specific_target="APROG_PROG_STATUS")
    for i in range(10):
        feature_mask = Model.get_random_mask(train_features.shape[1], probability=.15)
        train_features_subset = train_features[train_features.columns[feature_mask]]
        test_features_subset = test_features[test_features.columns[feature_mask]]
        #print(train_features_subset)
        print(test_features_subset.shape[1]," Features chosen ", train_features_subset.columns.values)
        clf = RandomForestClassifier(n_jobs=100, n_estimators=1000) #, max_depth=10)
        clf = clf.fit(train_features_subset, train_target)
        analysis = Analysis(test_features, test_target)
        print(analysis.important_features(clf, train_features_subset))
        print(clf.score(test_features_subset, test_target))
        # get feature set

        # create randomforest

        # fit forest

        # score forest



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