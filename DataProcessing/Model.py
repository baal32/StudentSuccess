import numpy as np
import pandas as pd

class Model(object):

    def __init__(self):
        pass

    @staticmethod
    def get_random_mask(mask_length, probability=.2):
        return pd.DataFrame(np.random.choice([False,True],mask_length,p=[1-probability, probability]))[0]

