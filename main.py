from DataCollection.DataSource import DataSource
from DataProcessing.Preprocessor import Processor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging
import csv

def one_hot(df, col_list):
    step_1 = df
    for col in col_list:
        just_dummies = pd.get_dummies(df[col], prefix=col+"_")
        step_1 = pd.concat([step_1, just_dummies], axis=1)
        step_1.drop([col], inplace=True, axis=1)
    return step_1

def main():
    """ Run the main program"""
    logger = logging.getLogger(__name__)
    #extract data
    full_frame = DataSource.get_all_data()

    #preprocess data
    preprocessor = Processor(full_frame)
    full_frame = preprocessor.one_hot()
    full_frame = preprocessor.drop_columns()
    full_frame = preprocessor.drop_rows_with_NA()


    #split into train and test
    train,test = preprocessor.split_train_test(.75)

    train_features, train_target, test_features, test_target = preprocessor.split_test_train_features_targets(.75, specific_target ="APROG_PROG_STATUS")

    #for non numeric features encode one-hot, dictvectorizer or pandas

    train_features = train.drop(outcomes,axis=1)
    test_features = test.drop(outcomes,axis=1)

    #  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

   # test_emplids = test["EMPLID"]
    train_outcomes = train[["APROG_PROG_STATUS","RETAIN_1_YEAR","RETAIN_2_YEAR","RETAIN_3_YEAR","GRADUATE_4_YEARS","GRADUATE_5_YEARS","GRADUATE_6_YEARS"]]
    test_outcomes = test[["APROG_PROG_STATUS","RETAIN_1_YEAR","RETAIN_2_YEAR","RETAIN_3_YEAR","GRADUATE_4_YEARS","GRADUATE_5_YEARS","GRADUATE_6_YEARS"]]

    train_target = train_outcomes["APROG_PROG_STATUS"]
    test_target = test_outcomes["APROG_PROG_STATUS"]
    #train_features = one_hot(train_features, {"GE_CRITICAL_THINKING_STATUS","GE_ENGLISH_COMPOSITION_STATUS","GE_ORAL_COMMUNICATIONS_STATUS","GE_MATH_STATUS","COLLEGE", "DEPARTMENT", "GENDER_DESC", "ETHNICITY_GRP_IPA", "MARITAL_STATUS", "HOME_COUNTRY"})
    #print(train_features)

    clf = RandomForestClassifier(n_jobs=10, n_estimators=100)
    clf = clf.fit(train_features,train_target)

    print(clf)

    print(clf.score(test_features,test_target))

    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), train_features.columns.values), reverse=True))


    loc_submission = "test.csv"
    with open(loc_submission, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Id,Outcome"])
        for e, val in enumerate(list(clf.predict(test_features))):
            writer.writerow([test_emplids[e],val])
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