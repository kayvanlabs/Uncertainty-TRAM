import os
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger_utils import setup_logging_config
from main_utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, get_run_id, resume=None, test_only=False,
                 modification=None, save=True, run_id_timestamp=False):

        """
        class to parse configuration json file. Handles hyperparameters for training, 
            initializations of modules, checkpoint saving and logging module.
        Inputs:
            config <python dict>: Dict containing configurations, hyperparameters for training,
                contents of `config.json` file for example.
            get_run_id <python function>: Function to get run_id from config dict.
            resume <python string>: String, path to the checkpoint being loaded.
            modification <python dict>: Dict keychain:value, specifying position values 
                to be replaced from config dict.
            run_id_timestamp <python bool>: Bool, if True, run_id will be timestamp.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.test_only = test_only
        
        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id_timestamp: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        elif "task" in self.config:
            run_id = self.config["task"]
        else:
            run_id = get_run_id(self.config)
            if run_id is None:
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir = save_dir / exper_name / run_id / 'models'
        self._log_dir = save_dir  / exper_name / run_id / 'log'

        # make directory for saving checkpoints and log.
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        if save:
            write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging_config(self.log_dir)

    @classmethod
    def from_args(cls, get_id_fun, args, options='', save_config=True):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))
        
        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        try: 
            test_only = args.test_only
        except AttributeError:
            test_only = False
        return cls(config, get_id_fun, resume, test_only, modification, save=save_config)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.
        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)