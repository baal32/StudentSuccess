import pyodbc
import pandas
from config import cfg

class DataSource(object):


    conn = None

    def __init__(self):
        pass

    def connect_SQL(self):
        self.conn = pyodbc.connect(cfg['db']['connections']['mssql']) #r'DRIVER={SQL Server Native Client 11.0};SERVER=.\MASTER;DATABASE=RADAR_DW;UID=python;PWD=python')

    def connect_Oracle(self):
        self.conn = pyodbc.connect(cfg['db']['connections']['oracle']) #pyodbc.connect('DSN=STUDENTSUCCESS;PWD=student_success')

    def query(self, query_string):
        return pandas.read_sql(query_string, self.conn)

    def get_all_data(self):
        self.connect_Oracle()
        return(self.query(cfg['db']['queries']['complete'])) #"select * from student_success.complete_vw@SM_SMDW_LINK where "                                 # "rownum < 50 AND "                                   "APROG_PROG_STATUS IN ('DC','CM') AND ACAD_PROG='UBACH'"))

    def cache(self):
        pass