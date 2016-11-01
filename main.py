import cProfile
import logging

import time
from sklearn.model_selection import cross_val_score

import config
from DataAnalysis.Analysis import Analysis
from DataAnalysis.Results import Results
from DataCollection.DataSource import DataSource
from DataProcessing.Classifier import SVMClassifier, RFClassifier, KNClassifier, ETClassifier, LinearSVCClassifier
from DataProcessing.Population import Population
from DataProcessing.Preprocessor import Processor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


logger = logging.getLogger(__name__)

def print_scores(y_test, y_predict, target_column):
    accuracy = ["Accuracy: ", accuracy_score(y_test[target_column], y_predict)]
    f1 = ["F1 Score: ", f1_score(y_test[target_column], y_predict, pos_label=1)]
    precision = ["Precision: ", precision_score(y_test[target_column], y_predict, pos_label=1)]
    recall = ["Recall: ",recall_score(y_test[target_column], y_predict, pos_label=1)]
    matthews = ["Matthews Coefficient: ", matthews_corrcoef(y_test[target_column], y_predict)]
    confusion = ["Confusion Matrix: ", confusion_matrix(y_test[target_column], y_predict)]
    scores = [accuracy, f1, precision, recall, matthews, confusion]
    logger.info("%s", scores)
    logger.info("%s", classification_report)
    #for score in scores:
    #    logger.info("%s %s", score[0], score[1])



def run_experiment(preprocessor, classifier_name, target_column, genetic_iterations, population_size):

    config.logger.info("\n\n\n\n\n\n\n\n\n\n\n\n\nStarting experiment using %s classifier*********************************", preprocessor.__class__)
    X_train, X_test, y_train, y_test = preprocessor.split_test_train(test_pct=0.25)

    #print("Train set:", X_train.columns)
    model = Population()



    retain_best = 1
    low_score_purge_pct = .5
    initial_population_chance_bit_on = .05

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
            child_id = str(iteration)+"-"+str(i)
            logger.debug("Child: %s", child_id)

            classifier = classifier_name()

            analysis = Analysis(X_test, y_test)

            X_train_feature_subset = X_train[X_train.columns[p]]
            X_test_feature_subset = X_test[X_test.columns[p]]

            #logger.info("X_train_feature_subset: %s", X_train_feature_subset.columns.values)



            # randomized grid search
            sqrtfeat = int(np.sqrt(X_train_feature_subset.shape[1]))
            param_grid = {"n_estimators": [10, 20, 30],
                          "criterion": ["gini", "entropy"],
                          "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
                          "max_depth": [5, 10, 25],
                          "min_samples_split": [2, 5, 10]}

            #param_grid = {"n_estimators": [1, 2, 3],
            #              "criterion": ["gini", "entropy"],
            #              "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
            #              "max_depth": [5, 7, 8],
            #             "min_samples_split": [2, 5, 10]}
            #print(param_grid)
            # create and fit a ridge regression model, testing random alpha values

            rsearch = RandomizedSearchCV(estimator=classifier.classifier, param_distributions=param_grid, n_iter=5)
            rsearch.fit(X_train_feature_subset, y_train[target_column])
            logger.info("Score: %f, Parameters: %s",rsearch.best_score_, rsearch.best_params_)
            #print(rsearch.best_estimator_)
            # end randomizzed grid search

            # apply tuned parameters to classifier
            classifier.classifier.set_params(**rsearch.best_params_)

            classifier.fit(X_train_feature_subset, y_train[target_column])


#            y_predict_proba = classifier.predict_proba(X_test_feature_subset)
            #y_predict = rsearch.predict(X_test_feature_subset)
            #score = rsearch.score(X_test_feature_subset, y_test[target_column])

            y_predict = classifier.predict(X_test_feature_subset)
            score = classifier.score(X_test_feature_subset, y_test[target_column])


            #score = classifier.cross_val_score(train_features_subset,train_target)
            #print(classifier.cross_val_score(X_train_feature_subset,y_train[target_column]))
            model.add_results(score, p, classifier, rsearch.best_params_, child_id)
            #model.add_results(score, p, classifier, None, child_id)
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
        logger.info("Experiment #%d Best score: %.4f Child: %s  Parameters: %s",
                    iteration, experiment_best.score, experiment_best.child_id,experiment_best.trained_classifier.get_params() )

        #logger.info("Cross validated score: %s",classifier.cross_val_score(X_train_feature_subset, y_train[target_column]))

        #print_scores(y_test, y_predict, target_column)


        #logger.info("Accuracy: %f F1: %s Precision: %f Recall: %f", accuracy_score(y_test[target_column], y_predict),f1_score(y_test[target_column], y_predict, pos_label=1),precision_score(y_test[target_column], y_predict, pos_label=1),recall_score(y_test[target_column], y_predict, pos_label=1))

        logger.info("Top features (%d): %s",experiment_best.feature_set.sum(),
                   experiment_best.trained_classifier.important_features(experiment_best.feature_set[experiment_best.feature_set].index))

        #experiment_best.trained_classifier.important_features(experiment_best.feature_set[experiment_best.feature_set].index)



        ''''# cross val score
        cross_score = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X, preprocessor.y[target_column], cv=5)
        cross_score2 = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X[preprocessor.X.columns[experiment_best.feature_set]], preprocessor.y[target_column], cv=5)
        base_score = experiment_best.trained_classifier.classifier.score(X_test[X_test.columns[experiment_best.feature_set]], y_test[target_column])
        logger.info("Cross val score: %0.2f (+/- %0.2f)   Cross val subset score: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2, cross_score2.mean(), cross_score2.std() * 2))
        '''
        result_frame.add_result("accuracy",experiment_best.score,iteration,target_column)
        #result_frame.add_result("cross_val_accuracy",             #cross val score


        # evolve children from remaining population
        population = model.evolve_children(population_size)

        #logger.info("Global best score: %f Features: %s", model.global_best[0].score,
         #           model.global_best[0].feature_set[0:3])
        #model.print_best_results(retain_best, analysis)

    # global score
    print("Global best *************************************************************************************")
    global_experiment_best = model.global_best[0]
    y_predict_global = global_experiment_best.trained_classifier.predict(X_test[X_test.columns[global_experiment_best.feature_set]])
    print_scores(y_test, y_predict_global, target_column)
    global_experiment_best.trained_classifier.important_features(global_experiment_best.feature_set[global_experiment_best.feature_set].index)
    return result_frame


def main():
    """ Run the main program"""


    full_frame = DataSource().get_all_data()
    Analysis.basic_stats(full_frame)
    target_column = "APROG_PROG_STATUS"
    preprocessor = Processor(full_frame)
    preprocessor.numeric_label_encoder()
    preprocessor.split_features_targets(target_cols = None)


    # perform one-hot on categorical features, fill NAs, drop columns we don't want to include right now
    # operations all reliant on config.yaml
    preprocessor.prepare_features()

    #preprocessor.remove_nonvariant_features()
    #full_frame = preprocessor.drop_rows_with_NA()


    print(Analysis.column_correlation(preprocessor.X, preprocessor.y[target_column]))

    # classifier = SVMClassifier()
    # classifier = RFClassifier()
    # classifier = ETClassifier()
    # classifier = KNClassifier()

    # classifiers = [RFClassifier, ETClassifier, KNClassifier]
    classifiers = [RFClassifier]

    #Analysis.basic_stats(train_features)
    genetic_iterations = 2
    population_size = 6
    for classifier in classifiers:
        result_frame = run_experiment(preprocessor, classifier, target_column, genetic_iterations, population_size)

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