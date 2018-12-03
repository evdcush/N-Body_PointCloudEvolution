import argparse
import yaml
import os

class AttrDict(dict):
    """ simply a dict accessed/mutated by attribute instead of index
    WARNING: cannot be pickled like normal dict/object
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

####### TODO
# dirttttttttyyyy
config_path = "/".join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1] + ['configs/config.yml'])

def parse():
    with open(config_path) as yml:
        config = AttrDict(yaml.load(yml))
    return config

# OLD PARSER
'''
class Parser:
    """ Wrapper for argparse parser
    """
    def __init__(self):
        self.p = argparse.ArgumentParser()
        self.add_parse_args()

    def add_parse_args(self,):
        add = self.p.add_argument

        # ==== Data variables
        add('--seed',         '-s', type=int, default=PARAMS_SEED)
        add('--rs_idx',       '-z', type=int, default=[10,19], nargs='+')
        add('--model_tag',    '-m', type=str, default='')
        add('--dataset_type', '-d', type=str, default='uni')

        # ==== Model parameter variables
        add('--graph_var',  '-k', type=int, default=14)

        # ==== Training variables
        add('--num_test',   '-t', type=int, default=NUM_VAL_SAMPLES)
        add('--num_iters',  '-i', type=int, default=2000)
        add('--batch_size', '-b', type=int, default=4)
        add('--restore',    '-r', action='store_true')
        add('--wr_meta',    '-w', action='store_true') # always write meta graph
        #add('--checkpoint', '-h', type=int, default=100)

    def parse_args(self):
        parsed = self.add_interpreted_args(AttrDict(vars(self.p.parse_args())))
        self.args = parsed
        return parsed

    def add_interpreted_args(self, parsed):
        # ==== redshifts
        redshift_idx = parsed.rs_idx
        redshifts = [REDSHIFTS[z] for z in redshift_idx]
        parsed.redshifts = redshifts

        # ==== Model-type
        mtype = SINGLE_STEP
        cat_rs = False
        if len(redshift_idx) > 2:
            mtype = MULTI_STEP
            cat_rs = True
        parsed.model_type = mtype
        parsed.cat_rs = cat_rs

        # ==== var_scope formatting
        vscope = parsed.var_scope
        parsed.var_scope = vscope.format(redshift_idx[0], redshift_idx[-1])
        return parsed


    def print_args(self):
        print('SESSION CONFIG\n{}'.format('='*79))
        margin = len(max(self.args, key=len)) + 1
        for k,v in self.args.items():
            print('{:>{margin}}: {}'.format(k,v, margin=margin))

'''
