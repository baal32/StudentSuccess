import pandas as pd
import numpy as np
from config import cfg

class Processor(object):

    df = None

    def __init__(self, df):
        self.df = df


    def list_cols_with_null_values(self):
        self.df.columns[pd.isnull(self.df).sum() > 0].tolist()


    def one_hot(self, col_list=None):
        if col_list is None:
            col_list = cfg['col_lists']['one_hot_list']
        step_1 = self.df
        for col in col_list:
            just_dummies = pd.get_dummies(self.df[col], prefix=col + "_")
            step_1 = pd.concat([step_1, just_dummies], axis=1)
            step_1.drop([col], inplace=True, axis=1)
        self.df = step_1
        return self.df

    def drop_columns(self, col_list=None):
        if col_list is None:
            col_list = cfg['col_lists']['drop_col_list']
        return self.df.drop(col_list, axis=1, inplace=True)

    def drop_rows_with_NA(self):
        return self.df.dropna(inplace=True)

    def split_test_train_features_targets(self,pct=.75, target_cols=None, specific_target=None):
        if target_cols is None:
            target_cols = cfg['col_lists']['outcome_list']
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
        self.df.fillna(value=0, inplace=True)
