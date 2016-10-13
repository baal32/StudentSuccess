from DataCollection.DataSource import DataSource
from DataProcessing.Preprocessor import Processor
from DataAnalysis.Plotter import Plotter
from DataAnalysis.Analysis import Analysis
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

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
    full_frame = preprocessor.drop_rows_with_NA()


    #split into train and test
    train_features, train_target, test_features, test_target = preprocessor.split_test_train_features_targets(.75, specific_target ="RETAIN_1_YEAR")

    print(train_features.columns[0:7])
    clf = RandomForestClassifier(n_jobs=10, n_estimators=100)
    clf = clf.fit(train_features,train_target)

    #Plotter.histogram(clf,train_features)
    print(clf)

    print(clf.score(test_features,test_target))

    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), train_features.columns.values), reverse=True))

    analysis = Analysis(test_features, test_target)
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

main()