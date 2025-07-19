import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
import json
from collections import OrderedDict

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        Class to parse configuration json file.
        """
        self._config = _update_config(config, modification)
        self.resume = resume

        save_dir = Path(self.config['trainer']['checkpoint_dir'])
        exper_name = self.config['name']
        
        # The main save directory is just the checkpoint dir.
        self._save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save the config file directly in the checkpoint directory with a descriptive name.
        config_save_path = self.save_dir / f"{exper_name}_config.json"
        write_json(self.config, config_save_path)

        # Log directory can still use a timestamp to differentiate logs.
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._log_dir = self.save_dir / 'log' / exper_name / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from cli arguments. Used in train, test.
        """
        if options:
            for opt in options:
                kwargs = opt.kwargs or {}
                args.add_argument(*opt.flags, default=None, type=opt.type, **kwargs)

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        # --- FIX: Prioritize the -c/--config argument. ---
        # The logic to find the correct config file is now handled by the calling scripts (train.py, test.py).
        msg_no_cfg = "Configuration file must be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        
        resume = Path(args.resume) if args.resume else None
        cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        
        # This part is for fine-tuning where you might provide a new config
        # to override the one from the resumed checkpoint's directory.
        if args.config and resume:
            if Path(args.config) != cfg_fname:
                 config.update(read_json(Path(args.config)))

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    @property
    def config(self):
        return self._config
    @property
    def save_dir(self):
        return self._save_dir
    @property
    def log_dir(self):
        return self._log_dir

# helper functions
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
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    return reduce(getitem, keys, tree)
