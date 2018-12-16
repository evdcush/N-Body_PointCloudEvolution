import os
import argparse
import yaml


####### TODO
# find solution for pathing, this is dirty
project_dir = os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]
config_path = "/".join(project_dir + ['configs/config.yml'])
HOME = os.environ['HOME']

class Parser:
    def __init__(self, config_path=config_path):
        self.argparser = argparse.ArgumentParser()
        self.config_path = config_path
        self.process_config()

    def process_config(self):
        _home_abspath = lambda v: v if not isinstance(v, str) else v.replace('~', HOME)
        with open(self.config_path) as conf:
            config = yaml.load(conf)
            for key, val in config.items():
                val = _home_abspath(val)
                kwargs = dict(default=val)
                if isinstance(val, int):
                    kwargs['type'] = int
                self.argparser.add_argument(f'--{key}', **kwargs)

    def parse(self):
        self.args = self.argparser.parse_args()
        return self.args
