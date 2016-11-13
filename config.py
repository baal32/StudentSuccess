import yaml
import logging

# get configuration file
with open("config.yaml") as ymlfile:
    try:
        cfg = yaml.load(ymlfile)['configuration']

    except yaml.YAMLError as exc:
        print(exc)

# get logger
#FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logging.basicConfig(level=logging.DEBUG, filemode='w')
logger = logging.getLogger()

fh = logging.FileHandler('output.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

