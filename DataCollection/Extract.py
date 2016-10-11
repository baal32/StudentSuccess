import pandas


class DataExtractor(object):

    data_frame = None
    datasource = None

    def __init__(self, datasource):
        self.datasource = datasource

    def connect_to_datasource(self):
        raise NotImplementedError

    def extract_data():
        # connect to database

        # extract data

        #

class OracleDataExtractor(DataExtractor):
    def connect_to_datasource(self):
        pass

