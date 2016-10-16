import pyodbc
import pandas as pd
from config import cfg
import os
import logging

class DataSource(object):


    conn = None
    df = None

    def __init__(self):
        pass

    def connect_SQL(self):
        self.conn = pyodbc.connect(cfg['db']['connections']['mssql']) #r'DRIVER={SQL Server Native Client 11.0};SERVER=.\MASTER;DATABASE=RADAR_DW;UID=python;PWD=python')

    def connect_Oracle(self):
        self.conn = pyodbc.connect(cfg['db']['connections']['oracle']) #pyodbc.connect('DSN=STUDENTSUCCESS;PWD=student_success')

    def query(self, query_string):
        return pd.read_sql(query_string, self.conn)

    def get_all_data(self, read_from_cache=True, filename = None):
        if filename is None:
            filename = cfg['db']['cache']
        if os.path.exists(filename):
            self.read_from_cache(filename)
        else:
            self.connect_Oracle()
            self.df = self.query(cfg['db']['queries']['complete'])
            self.write_to_cache(filename)
        return self.df

    def write_to_cache(self, filename=None):
        if filename is None:
            filename = cfg['db']['cache']
        self.df.to_pickle(filename)

    def read_from_cache(self, filename = None):
        if filename is None:
            filename = cfg['db']['cache']
        self.df = pd.read_pickle(filename)
