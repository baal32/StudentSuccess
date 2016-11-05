from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

class Processor(object):

    logger = config.logger
    df = None
    X = None
    y = None

    def __init__(self, df):
        self.df = df


    # TODO not used unless it will return dataframe
    def remove_nonvariant_features(self):
        self.logger.info('Crossvalidation started... ')
        selector = VarianceThreshold()
        selector.fit(self.X)
        self.logger.info('Number of features used... ' + str(Counter(selector.get_support())[True]))
        self.logger.info('Number of features ignored... ' + str(Counter(selector.get_support())[False]))
        self.X = selector.transform(self.X)
        return self.X
        #return selector

    def numeric_label_encoder(self, encode_columns = None):
        if encode_columns is None:
            encode_columns = config.cfg['col_lists']['label_encode_list']
        le = LabelEncoder()
        self.df[encode_columns] = le.fit_transform(self.df[encode_columns])
        pass

    def list_cols_with_null_values(self):
        self.df.columns[pd.isnull(self.df).sum() > 0].tolist()

    def list_cols_non_numeric(self):
        print(self.X.select_dtypes(exclude=[np.number]).columns)

    def one_hot(self, col_list=None):
        if col_list is None:
            col_list = config.cfg['col_lists']['one_hot_list']
        step_1 = self.X
        for col in col_list:
            just_dummies = pd.get_dummies(self.X[col], prefix=col + "_")
            step_1 = pd.concat([step_1, just_dummies], axis=1)
            step_1.drop([col], inplace=True, axis=1)
        self.X = step_1
        return self.X

    def drop_columns(self, col_list=None):
        if col_list is None:
            col_list = config.cfg['col_lists']['drop_col_list']
        self.X.drop(col_list, axis=1, inplace=True)
        return self.X

    def drop_rows_with_NA(self):
        self.X.dropna(inplace=True)
        return self.X

    def split_features_targets(self, target_cols):
        if target_cols is None:
            target_cols = config.cfg['col_lists']['outcome_list']
        X = self.df.drop(target_cols, axis=1)
        y = self.df[target_cols]
        self.X = X
        self.y = y
        return X,y

    def split_test_train(self, test_pct = .25):
        msk = np.random.randn(len(self.df)) > test_pct
        X_train, y_train = self.X[msk], self.y[msk]
        X_test, y_test = self.X[~msk], self.y[~msk]
        return X_train, X_test, y_train, y_test

    def split_test_train_features_targets(self,pct=.75, target_cols=None, specific_target=None):
        # TODO delete if no longer used
        if target_cols is None:
            target_cols = config.cfg['col_lists']['outcome_list']
        self.df['is_train'] = np.random.uniform(0,1,len(self.df)) <= pct
        train,test = self.df[self.df['is_train'] == True], self.df[self.df['is_train'] == False]
        train_features = train.drop(target_cols, axis=1)
        test_features = test.drop(target_cols, axis=1)
        train_target = train[specific_target]
        test_target = test[specific_target]
        return train_features, train_target, test_features, test_target

    def impute_missing_values(self):
        # TODO fix this garbage
        col_list = ["SAT_VERBAL","SAT_MATH","HIP_GEL","HIP_EMPLOYMENT","HIP_CAMP","HIP_EARLYSTART","HIP_EOP"]
        self.X.fillna(value=0, inplace=True)
        return self.X

    def impute_values(self, column, condition, imputation_algorithm="mean"):
        df = self.X
        if imputation_algorithm == "mean":
            df.loc[((df[column] ==  condition) | df[column].isnull()),[column]] = df[df[column] != 0][column].mean()
        pass


    def train_test_split(self, test_size=0.25, random_state=42):
        self.logger.info("Splitting test and train with %f%% test data")
        return train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)

    def prepare_features(self, target_columns = None):
        # TODO self.numeric_label_encoder(target_columns)
        self.one_hot()
        self.drop_columns()
        self.impute_missing_values()
        # TODO make this more flexible
        [self.impute_values(column, 0, "mean") for column in config.cfg['col_lists']['impute_columns']]
