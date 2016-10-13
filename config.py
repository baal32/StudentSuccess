import yaml

#with open("config.yml", 'r') as ymlfile:
with open("config.yaml") as ymlfile:
    try:
        cfg = yaml.load(ymlfile)['configuration']

    except yaml.YAMLError as exc:
        print(exc)