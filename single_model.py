import sys
import subprocess
import json

from omegaconf import OmegaConf
from subprocess import PIPE, run

if __name__ == '__main__':
    overrides = OmegaConf.to_container(OmegaConf.from_cli())
    overrides = dict((key.replace('--',''), value) for (key, value) in overrides.items())
    data_number = overrides.pop('data_number')
    config_path = overrides.pop('config_path')
    override_args = ('--overrides', json.dumps(overrides)) if overrides else []
    print(override_args)
    
    with open(config_path, "r") as jsonfile:
        config_dict = json.load(jsonfile)

    if "GAT" in config_dict:
        if config_dict["GAT"] == True:
            result = run(f'python gat.py {config_path} {data_number}', 
                            stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    else:
        result = run(f'python gcn.py {config_path} {data_number}', 
                        stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)

    print(result.returncode, result.stdout, result.stderr)
    
    
