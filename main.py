import cProfile
import logging
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

import config
from DataAnalysis.Analysis import Analysis
from DataAnalysis.Results import Results
from DataCollection.DataSource import DataSource
from DataProcessing.Population import Population
from DataProcessing.Preprocessor import Processor

logger = logging.getLogger(__name__)




def run_predictions(full_frame, global_best_model, X_test, y_test, target_column):
    X_test_subset = X_test[X_test.columns[global_best_model.feature_set]]
    y_test_target = y_test[target_column]
    clf = global_best_model.trained_classifier.classifier
    analysis = Analysis(X_test_subset,y_test_target)
    analysis.interpret_tree(clf,X_test_subset.iloc[0:100], full_frame)
    #for r in X_test_subset.iterrows():
     #   clf.predict(r, )
    pass


def run_experiment(full_frame, preprocessor, classifier_name, target_column, genetic_iterations, population_size):
    custom_scorer = make_scorer(matthews_corrcoef)
    config.logger.info("\n\n\n\n%s Starting experiment using %s classifier*********************************", target_column, preprocessor.__class__)
    X_train, X_test, y_train, y_test = preprocessor.split_test_train(test_pct=0.15)

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
            #logger.debug("Child: %s", child_id)

            classifier = classifier_name(n_estimators=25)

            # Apply population mask to feature set
            X_train_feature_subset = X_train[X_train.columns[feature_mask]]
            X_test_feature_subset = X_test[X_test.columns[feature_mask]]

            #logger.info("X_train_feature_subset: %s", X_train_feature_subset.columns.values)

            # Tune hyperparameters
            sqrtfeat = int(np.sqrt(X_train_feature_subset.shape[1]))
            param_grid = {"criterion": ["gini", "entropy"],
                          "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
                          "max_depth": [3, 6, 9],
                          "min_samples_split": [2, 5, 10]}

            #param_grid = {"C": [1,2,3],
            #              "loss": ["hinge","squared_hinge"],
            #              "max_iter": [100, 500, 1000]}
            #param_grid = {"n_estimators": [1, 2, 3],
            #              "criterion": ["gini", "entropy"],
            #              "max_features": [sqrtfeat - 1, sqrtfeat, sqrtfeat + 1],
            #              "max_depth": [5, 7, 8],
            #             "min_samples_split": [2, 5, 10]}




            parameter_random_search = RandomizedSearchCV(scoring = custom_scorer, estimator=classifier, param_distributions=param_grid, n_iter=5)
            parameter_random_search.fit(X_train_feature_subset, y_train[target_column])
            random_search_best_score = parameter_random_search.best_score_
            random_search_best_params = parameter_random_search.best_params_
            random_search_best_estimator = parameter_random_search.best_estimator_
            model.add_results(random_search_best_score, feature_mask, random_search_best_estimator, random_search_best_params, child_id)

            #logger.debug("%s Random parameter score: %f, Parameters: %s, \nFeatures: %s",child_id,parameter_random_search.best_score_, parameter_random_search.best_params_, X_train_feature_subset.columns)
            #print(rsearch.best_estimator_)
            # end randomizzed grid search

            # apply tuned parameters to classifier
            #classifier.set_params(**parameter_random_search.best_params_)
            #classifier.fit(X_train_feature_subset, y_train[target_column])
            #y_pred = classifier.predict(X_test_feature_subset)

            #score = classifier.score(X_test_feature_subset, y_test[target_column])
            #score = custom_scorer(classifier,X_test_feature_subset, y_test[target_column])

            # score without test data to prevent information leakage into pipeline
            #score = custom_scorer(classifier,X_train_feature_subset, y_train[target_column])

            # score against test data for results dataframe
            #test_score = custom_scorer(classifier,X_test_feature_subset, y_test[target_column])
            #score = fucking_scorer(classifier, y_test[target_column], y_pred)


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
        for score in Analysis.classification_scores(y_test, y_predict_experiment_best, target_column):
            #logger.debug("%s, %s", score[0], score[1])
            result_frame.add_result(score[0], score[1], iteration, target_column)

        #logger.debug("Generation #%d Highest  training score: %.4f Child: %s  Parameters: %s", iteration, generation_best_model.score, generation_best_model.child_id,generation_best_model.trained_classifier.get_params() )


        #logger.info("Cross validated score: %s",classifier.cross_val_score(X_train_feature_subset, y_train[target_column]))




        ''''# cross val score
        cross_score = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X, preprocessor.y[target_column], cv=5)
        cross_score2 = cross_val_score(experiment_best.trained_classifier.classifier, preprocessor.X[preprocessor.X.columns[experiment_best.feature_set]], preprocessor.y[target_column], cv=5)
        base_score = experiment_best.trained_classifier.classifier.score(X_test[X_test.columns[experiment_best.feature_set]], y_test[target_column])
        logger.info("Cross val score: %0.2f (+/- %0.2f)   Cross val subset score: %0.2f (+/- %0.2f)" % (cross_score.mean(), cross_score.std() * 2, cross_score2.mean(), cross_score2.std() * 2))
        '''

    # global score

    print("\n\n\n\n\nAnalysis of final results\nGlobal best *************************************************************************************")
    global_best_model = model.global_best[0]

    y_predict_global = global_best_model.trained_classifier.predict(X_test[X_test.columns[global_best_model.feature_set]])

    #print("Sanity check score %f" % global_best_model.trained_classifier.score(X_test[X_test.columns[global_best_model.feature_set]], y_test[target_column]))

    logger.debug("%s Scores******************************", target_column)
    for score in Analysis.classification_scores(y_test, y_predict_global, target_column):
        logger.debug("%s, %s", score[0], score[1])
        #result_frame.add_result(score[0], score[1], iteration, target_column)

    importance_threshold = 0.05
    important_features = [t[1] for t in Analysis.important_features(global_best_model.trained_classifier,global_best_model.feature_set[global_best_model.feature_set].index, target_column, threshold=importance_threshold) if float(t[0]) > importance_threshold]
    logger.debug("Top features (%d): %s", global_best_model.feature_set.sum(),important_features)
    logger.debug("Top hyperparameters: %s, %s", global_best_model.trained_classifier.get_params(), global_best_model.classifier_params)
    #if type(classifier) != LinearSVCClassifier:
    result_frame.plot_roc(global_best_model.trained_classifier, X_test[X_test.columns[global_best_model.feature_set]], y_test[target_column])
    logger.debug("%s",Analysis.classification_scores(y_test, y_predict_global, target_column))
    #global_best_model.trained_classifier.important_features(global_best_model.feature_set[global_best_model.feature_set].index, target_column)

    #for feature_column,_ in global_best_model.feature_set[global_best_model.feature_set].iteritems():
    #    Analysis.agg_by_target(preprocessor.X[feature_column], preprocessor.y[target_column],aggregation_method = 'AVG')

    Analysis.crosstab(y_test[target_column], y_predict_global)



    logger.info("\n\n\n\Running Predictions against test data")

    #run_predictions(full_frame, global_best_model, X_test, y_test, target_column)

    tree_depth=5
    final_classifier = DecisionTreeClassifier()
    final_classifier.set_params(**global_best_model.classifier_params)
    final_classifier.max_depth = tree_depth
    final_classifier.max_features = len(important_features)
    final_classifier.fit(X_train[important_features], y_train[target_column])
    logger.debug("Single decision tree score: %f" , final_classifier.score(X_test[important_features], y_test[target_column]))
    Results.plot_decision_tree(final_classifier, important_features, target_column, tree_depth)
    return result_frame


def main():
    """ Run the main program"""


    full_frame = DataSource().get_all_data()
    Analysis.basic_stats(full_frame)

    # targets for FTF
    # targets = ["GRADUATED", "WITHIN_2_YEARS", "WITHIN_3_YEARS", "WITHIN_4_YEARS"]
    targets = ["GRADUATED", "WITHIN_4_YEARS", "WITHIN_5_YEARS", "WITHIN_6_YEARS"]
    targets = ["GRADUATED",  "WITHIN_5_YEARS", "WITHIN_6_YEARS", "RETAIN_1_YEAR", "RETAIN_2_YEAR", "RETAIN_3_YEAR","WITHIN_4_YEARS"]
    #targets = ["GRADUATED", "WITHIN_2_YEARS", "WITHIN_3_YEARS", "WITHIN_4_YEARS"]

    # targets for transfers
    #targets = ["GRADUATED", "WITHIN_2_YEARS", "WITHIN_3_YEARS", "WITHIN_4_YEARS", "EXACT_2_YEARS", "EXACT_3_YEARS", "EXACT_4_YEARS"]

    #targets = ["GRADUATED"]



    preprocessor = Processor(full_frame)
    preprocessor.numeric_label_encoder()
    preprocessor.split_features_targets(target_cols = None)

    # perform one-hot on categorical features, fill NAs, drop columns we don't want to include right now
    # operations all reliant on config.yaml
    Analysis.nulls_by_feature(preprocessor.X)
    preprocessor.prepare_features()
    Analysis.nulls_by_feature(preprocessor.X)

    #preprocessor.remove_nonvariant_features()
    #full_frame = preprocessor.drop_rows_with_NA()




    # classifier = SVMClassifier()
    # classifier = RFClassifier()
    # classifier = ETClassifier()
    # classifier = KNClassifier()

    # classifiers = [RFClassifier, ETClassifier, KNClassifier]
    classifiers = [RandomForestClassifier]

    #Analysis.basic_stats(train_features)
    genetic_iterations = 10
    population_size = 10
    for classifier in classifiers:
        for target_column in targets:
            print(Analysis.column_correlation(preprocessor.X, preprocessor.y[target_column]))
            result_frame = run_experiment(full_frame, preprocessor, classifier, target_column, genetic_iterations, population_size)
            result_frame.plot_scores(['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC','Matthews Coefficient'])
#            result_frame.write_result(target_column + "_results_" + time.strftime("%d_%m_%Y_%H%M"))
            result_frame.write_result(target_column + "_results")
            print(result_frame.score_list)




    #plt.show()
    #analysis.interpret_tree(model.get_best_feature_sets(retain_best)[0].trained_classifier.classifier, X_test[X_test.columns[model.get_best_feature_sets(retain_best)[0].feature_set]][0:3])


cProfile.run(main())