import cProfile
import logging

import time
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import config
from DataAnalysis.Analysis import Analysis
from DataAnalysis.Results import Results
from DataCollection.DataSource import DataSource
from DataProcessing.Classifier import SVMClassifier, RFClassifier, KNClassifier, ETClassifier, LinearSVCClassifier
from DataProcessing.Population import Population
from DataProcessing.Preprocessor import Processor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


logger = logging.getLogger(__name__)

def classification_scores(y_test, y_predict, target_column):
    accuracy = ["Accuracy", accuracy_score(y_test[target_column], y_predict)]
    f1 = ["F1 Score", f1_score(y_test[target_column], y_predict, pos_label=1)]
    precision = ["Precision", precision_score(y_test[target_column], y_predict, pos_label=1)]
    recall = ["Recall",recall_score(y_test[target_column], y_predict, pos_label=1)]
    matthews = ["Matthews Coefficient", matthews_corrcoef(y_test[target_column], y_predict)]
    confusion = ["Confusion Matrix", confusion_matrix(y_test[target_column], y_predict)]
    roc = ["ROC Score", roc_auc_score(y_test[target_column], y_predict)]
    scores = [accuracy, f1, precision, recall, matthews, roc]
    #logger.info("%s", scores)
    #logger.info("%s", classification_report)
    return scores
    #for score in scores:
    #    logger.info("%s %s", score[0], score[1])


def score_model(global_best_model):
    pass


def run_predictions(full_frame, global_best_model, X_test, y_test, target_column):
    X_test_subset = X_test[X_test.columns[global_best_model.feature_set]]
    y_test_target = y_test[target_column]

    clf = global_best_model.trained_classifier.classifier

    analysis = Analysis(X_test_subset,y_test_target)
    analysis.interpret_tree(clf,X_test_subset, full_frame)
    #for r in X_test_subset.iterrows():
     #   clf.predict(r, )



    pass


def run_experiment(full_frame, preprocessor, classifier_name, target_column, genetic_iterations, population_size):


    config.logger.info("\n\n\n\nStarting experiment using %s classifier*********************************", preprocessor.__class__)
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
        for child_number,feature_mask  in population.iterrows():
            child_id = str(iteration)+"-"+str(child_number)
            logger.debug("Child: %s", child_id)

            classifier = classifier_name()

            #analysis = Analysis(X_test, y_test)

            # Apply population mask to feature set
            X_train_feature_subset = X_train[X_train.columns[feature_mask]]
            X_test_feature_subset = X_test[X_test.columns[feature_mask]]

            #logger.info("X_train_feature_subset: %s", X_train_feature_subset.columns.values)



            # Tune hyperparameters
            sqrtfeat = int(np.sqrt(X_train_feature_subset.shape[1]))
            param_grid = {"n_estimators": [10, 25, 50],
                          "criterion": ["gini", "entropy"],
                          "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
                          "max_depth": [5, 10, 25],
                          "min_samples_split": [2, 5, 10]}

            #param_grid = {"n_estimators": [1, 2, 3],
            #              "criterion": ["gini", "entropy"],
            #              "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
            #              "max_depth": [5, 7, 8],
            #             "min_samples_split": [2, 5, 10]}

            parameter_random_search = RandomizedSearchCV(estimator=classifier.classifier, param_distributions=param_grid, n_iter=5)
            parameter_random_search.fit(X_train_feature_subset, y_train[target_column])
            logger.info("%s Random parameter score: %f, Parameters: %s, \nFeatures: %s",child_id,parameter_random_search.best_score_, parameter_random_search.best_params_, X_train_feature_subset.columns)
            #print(rsearch.best_estimator_)
            # end randomizzed grid search

            # apply tuned parameters to classifier
            classifier.set_params(**parameter_random_search.best_params_)
            classifier.fit(X_train_feature_subset, y_train[target_column])
            score = classifier.score(X_test_feature_subset, y_test[target_column])
            model.add_results(score, feature_mask, classifier, parameter_random_search.best_params_, child_id)

#            y_predict_proba = classifier.predict_proba(X_test_feature_subset)
            #y_predict = rsearch.predict(X_test_feature_subset)
            #score = rsearch.score(X_test_feature_subset, y_test[target_column])

            #y_predict = classifier.predict(X_test_feature_subset)



            #score = classifier.cross_val_score(train_features_subset,train_target)
            #print(classifier.cross_val_score(X_train_feature_subset,y_train[target_column]))

            #model.add_results(score, p, classifier, None, child_id)
            # get feature set


        # sort scores, purge weakest of population
        model.process_results(low_score_purge_pct)
        population = model.evolve_children(population_size)

        # analyze best model for experiment
        generation_best_model = model.global_best[0]
        y_predict_experiment_best = generation_best_model.trained_classifier.predict(X_test[X_test.columns[generation_best_model.feature_set]])


        logger.info("Experiment #%d Best score: %.4f Child: %s  Parameters: %s",
                    iteration, generation_best_model.score, generation_best_model.child_id,generation_best_model.trained_classifier.get_params() )
        logger.info("Top features (%d): %s", generation_best_model.feature_set.sum(),
                    generation_best_model.trained_classifier.important_features(
                        generation_best_model.feature_set[generation_best_model.feature_set].index))
        for score in classification_scores(y_test, y_predict_experiment_best, target_column):
            result_frame.add_result(score[0],score[1],iteration,target_column)
        #logger.info("Cross validated score: %s",classifier.cross_val_score(X_train_feature_subset, y_train[target_column]))

        Results.plot_roc(generation_best_model.trained_classifier,X_test[X_test.columns[generation_best_model.feature_set]], y_test[target_column] )
        #print_scores(y_test, y_predict, target_column)


        #logger.info("Accuracy: %f F1: %s Precision: %f Recall: %f", accuracy_score(y_test[target_column], y_predict),f1_score(y_test[target_column], y_predict, pos_label=1),precision_score(y_test[target_column], y_predict, pos_label=1),recall_score(y_test[target_column], y_predict, pos_label=1))



        #experiment_best.trained_classifier.important_features(experiment_best.feature_set[experiment_best.feature_set].index)



        ''''# cross val score
        cross_score = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X, preprocessor.y[target_column], cv=5)
        cross_score2 = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X[preprocessor.X.columns[experiment_best.feature_set]], preprocessor.y[target_column], cv=5)
        base_score = experiment_best.trained_classifier.classifier.score(X_test[X_test.columns[experiment_best.feature_set]], y_test[target_column])
        logger.info("Cross val score: %0.2f (+/- %0.2f)   Cross val subset score: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2, cross_score2.mean(), cross_score2.std() * 2))
        '''

        #result_frame.add_result("cross_val_accuracy",             #cross val score


        # evolve children from remaining population


        #logger.info("Global best score: %f Features: %s", model.global_best[0].score,
         #           model.global_best[0].feature_set[0:3])
        #model.print_best_results(retain_best, analysis)

    # global score



    print("\n\n\n\n\nAnalysis of final results\nGlobal best *************************************************************************************")
    global_best_model = model.global_best[0]

    score_model(global_best_model)

    y_predict_global = global_best_model.trained_classifier.predict(X_test[X_test.columns[global_best_model.feature_set]])

    print("Sanity check score %f", global_best_model.trained_classifier.score(X_test[X_test.columns[global_best_model.feature_set]], y_test[target_column]))


    logger.info("%s",classification_scores(y_test, y_predict_global, target_column))
    global_best_model.trained_classifier.important_features(global_best_model.feature_set[global_best_model.feature_set].index)
    for feature_column,_ in global_best_model.feature_set[global_best_model.feature_set].iteritems():
        Analysis.agg_by_target(preprocessor.X[feature_column], preprocessor.y[target_column],aggregation_method = 'AVG')

    Analysis.crosstab(y_test[target_column], y_predict_global)



    logger.info("\n\n\n\Running Predictions against test data")

    run_predictions(full_frame, global_best_model, X_test, y_test, target_column)

    final_classifier = DecisionTreeClassifier()
    final_classifier.fit(X_train[X_train.columns[global_best_model.feature_set]], y_train[target_column])
    print("Final score: %f" % final_classifier.score(X_test[X_test.columns[global_best_model.feature_set]], y_test[target_column]))

    return result_frame


def main():
    """ Run the main program"""


    full_frame = DataSource().get_all_data()
    Analysis.basic_stats(full_frame)
    target_column = "GRADUATED"
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
    genetic_iterations = 10
    population_size = 10
    for classifier in classifiers:
        result_frame = run_experiment(full_frame, preprocessor, classifier, target_column, genetic_iterations, population_size)

    print(result_frame.score_list)
    result_frame.plot_scores(['Accuracy', 'F1 Score','Precision','Recall'])
    result_frame.write_result("results_"+time.strftime("%d_%m_%Y_%H%M"))

    #analysis.interpret_tree(model.get_best_feature_sets(retain_best)[0].trained_classifier.classifier, X_test[X_test.columns[model.get_best_feature_sets(retain_best)[0].feature_set]][0:3])


cProfile.run(main())