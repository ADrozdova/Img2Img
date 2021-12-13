import json
from datetime import datetime
from pathlib import Path
from collections import OrderedDict


class ConfigParser:
    def __init__(self, filepath):
        json_file = open(filepath, "r")
        self._config = json.load(json_file)
        json_file.close()

        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        run_id = datetime.now().strftime(r"%m%d_%H%M%S%f")
        self._save_dir = save_dir / "models" / exper_name / run_id
        self._log_dir = save_dir / "log" / exper_name / run_id

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, name):
        return self.config[name]

    def init_obj(self, obj_dict, module, *args, **kwargs):
        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @classmethod
    def from_args(cls, args):
        if not isinstance(args, tuple):
            args = args.parse_args()

        cfg_fname = Path(args.config)

        json_file = open(Path(cfg_fname), "r")
        config = json.load(json_file, object_hook=OrderedDict)
        json_file.close()

        return cls(config)

