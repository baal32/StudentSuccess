import cProfile
import logging

import time
from sklearn.model_selection import cross_val_score

import config
from DataAnalysis.Analysis import Analysis
from DataAnalysis.Results import Results
from DataCollection.DataSource import DataSource
from DataProcessing.Classifier import SVMClassifier, RFClassifier, KNClassifier, ETClassifier
from DataProcessing.Population import Population
from DataProcessing.Preprocessor import Processor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score
import numpy as np


logger = logging.getLogger(__name__)

def run_experiment(preprocessor, classifier_name, target_column, genetic_iterations):


    X_train, X_test, y_train, y_test = preprocessor.split_test_train(test_pct=0.25)

    #print("Train set:", X_train.columns)
    model = Population()


    population_size = 10
    retain_best = 3
    low_score_purge_pct = .5
    initial_population_chance_bit_on = .05

    estimators = 10
    jobs = 10
    #initial population randomly generated



    result_frame = Results(classifier_name)
    # experiments
    for iteration in range(genetic_iterations):

        config.logger.debug("Beginning experiment #%d", iteration)
        # create the population that will be analyzed
        if iteration == 0:
            population = model.get_random_mask((population_size, X_train.shape[1]), probability=initial_population_chance_bit_on, column_headers = X_train.columns.values )

        # for each experiment reset the score dataframe to be population by the newest population
        model.reset_fitness_scores_and_features()
        for i,p  in population.iterrows():
            classifier = classifier_name()

            child_id = str(iteration)+"-"+str(i)
            logger.debug("Child: %s", child_id)
            analysis = Analysis(X_test, y_test)



            X_train_feature_subset = X_train[X_train.columns[p]]
            #print(X_train_feature_subset.columns)
            X_test_feature_subset = X_test[X_test.columns[p]]
            classifier.fit(X_train_feature_subset, y_train[target_column])
            y_predict_proba = classifier.predict_proba(X_test_feature_subset)
            y_predict = classifier.predict(X_test_feature_subset)
            score = classifier.score(X_test_feature_subset, y_test[target_column])

            logger.debug("%s Accuracy: %f", child_id,accuracy_score(y_test[target_column], y_predict))
            logger.debug("%s F1: %f", child_id,f1_score(y_test[target_column], y_predict, pos_label='DC'))
            logger.debug("%s Precision: %f",child_id, precision_score(y_test[target_column], y_predict, pos_label='DC'))
            logger.debug("%s Recall: %f",child_id, recall_score(y_test[target_column], y_predict, pos_label='DC'))
            #logger.debug("%s ROC: %f", child_id,roc_auc_score(y_test[target_column], y_predict))
            #logger.debug("%s R2: %f", child_id,r2_score(y_test[target_column], y_predict))





            #score = classifier.cross_val_score(train_features_subset,train_target)
            #print(score)
            model.add_results(score, p, classifier, child_id)
            # get feature set


        model.append_global_best_to_models()
        # sort scores
        model.sort_results()
        # throw away lowest scorers (population_purge) by pct
        model.purge_low_scores(population_purge_pct=low_score_purge_pct)
        # take the rest of the scores, append the global best, and determine new global best
        model.evaluate_global_best()

        experiment_best = model.global_best[0]

        y_predict = experiment_best.trained_classifier.predict(X_test[X_test.columns[experiment_best.feature_set]])
        logger.info("Experiment #%d Best score: %.4f Child: %s",
                    iteration, experiment_best.score, experiment_best.child_id )
        logger.info("Accuracy: %f F1: %s Precision: %f Recall: %f", accuracy_score(y_test[target_column], y_predict),f1_score(y_test[target_column], y_predict, pos_label='DC'),precision_score(y_test[target_column], y_predict, pos_label='DC'),recall_score(y_test[target_column], y_predict, pos_label='DC'))
        logger.info("Top features (%d): %s",experiment_best.feature_set.sum(),
                   experiment_best.trained_classifier.important_features(experiment_best.feature_set[experiment_best.feature_set].index))


        # cross val score
        #cross_score = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X, preprocessor.y, cv=5)
        #cross_score2 = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X[preprocessor.X.columns[experiment_best.feature_set]], preprocessor.y, cv=5)
        #logger.info("Cross val score: %0.2f (+/- %0.2f)   Cross val subset score: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2, cross_score2.mean(), cross_score2.std() * 2))
        result_frame.add_result("accuracy",experiment_best.score,iteration,target_column)
        #result_frame.add_result("cross_val_accuracy",             #cross val score


        # evolve children from remaining population
        population = model.evolve_children(population_size)

        #logger.info("Global best score: %f Features: %s", model.global_best[0].score,
         #           model.global_best[0].feature_set[0:3])
        model.print_best_results(retain_best, analysis)
    return result_frame


def main():
    """ Run the main program"""


    full_frame = DataSource().get_all_data()
    target_column = "APROG_PROG_STATUS"
    preprocessor = Processor(full_frame)

    preprocessor.split_features_targets(target_cols = None)

    # perform one-hot on categorical features, fill NAs, drop columns we don't want to include right now
    # operations all reliant on config.yaml
    preprocessor.prepare_features()

    #preprocessor.remove_nonvariant_features()
    #full_frame = preprocessor.drop_rows_with_NA()

    Analysis.basic_stats(full_frame)
    print(Analysis.column_correlation(preprocessor.X, preprocessor.y[target_column]))

    # classifier = SVMClassifier()
    # classifier = RFClassifier()
    classifiers = [ETClassifier]
    # classifier = KNClassifier()

    #Analysis.basic_stats(train_features)
    genetic_iterations = 15

    for classifier in classifiers:
        result_frame = run_experiment(preprocessor, classifier, target_column, genetic_iterations)

    print(result_frame.score_list)
    result_frame.plot_scores()
    result_frame.write_result("results_"+time.strftime("%d_%m_%Y"))

    #analysis.interpret_tree(model.get_best_feature_sets(retain_best)[0].trained_classifier.classifier, X_test[X_test.columns[model.get_best_feature_sets(retain_best)[0].feature_set]][0:3])



    #Plotter.histogram(clf,train_features)
 #   print(clf)

  #  print(clf.score(test_features,test_target))

   # analysis = Analysis(test_features, test_target)
   # print(analysis.important_features(clf,train_features))

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


cProfile.run(main())