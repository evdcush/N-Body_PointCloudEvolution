import argparse
import yaml
import os

####### TODO
# find solution for pathing, this is dirty
project_dir = os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]
config_path = "/".join(project_dir + ['configs/config.yml'])

class Parser:
    def __init__(self, config_path=config_path):
        self.argparser = argparse.ArgumentParser()
        self.config_path = config_path
        self.process_config()

    def process_config(self):
        with open(self.config_path) as conf:
            config = yaml.load(conf)
            for key, val in config.items():
                kwargs = dict(default=val)
                if isinstance(val, int):
                    kwargs['type'] = int
                self.argparser.add_argument(f'--{key}', **kwargs)

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args
