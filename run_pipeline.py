import sys
import os

import yaml
import json

CONFIG_PATH = sys.argv[1]

USE_SLURM = True
if len(sys.argv) > 2:
    USE_SLURM = sys.argv[2]

with open(CONFIG_PATH, "r") as jsonfile:
    config_dict = json.load(jsonfile)
if 'RUN_LABEL_PROP' in config_dict:
    RUN_LABEL_PROP = config_dict['RUN_LABEL_PROP']
else:
    RUN_LABEL_PROP = True




# if not set otherwise, construct graphs

if 'PATH_TO_GRAPHS' not in config_dict:
    bash_command = f'python construct_graphs.py {CONFIG_PATH}'
    os.system(bash_command)




if USE_SLURM:
    with open('config/slurm_sweep_template.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    sweep_config['parameters']['config_path'] = {'values':[CONFIG_PATH]}
    sweep_config['parameters']['data_number'] = {'values':list(range(config_dict["NUM_SPLITS"]))}
    #sweep_config['parameters']['graph_type'] = {'values':config_dict["GRAPH_TYPES"]}
    sweep_config_path = CONFIG_PATH.split('.json')[0] + '_slurm_sweep.yaml'
    with open(sweep_config_path, 'w') as file:\
        final_sweep_config = yaml.dump(sweep_config, file)

    with open(CONFIG_PATH, "r") as jsonfile:
        config_dict = json.load(jsonfile)
    num_jobs = config_dict["NUM_SPLITS"] #* len(config_dict["GRAPH_TYPES"])
    
    bash_command = f'python single_model_job.py {sweep_config_path} {num_jobs}'
    os.system(bash_command)
else:
    bash_command = f'python gcn.py {CONFIG_PATH}'
    os.system(bash_command)

    bash_command = f'python second_stage.py {CONFIG_PATH}'
    os.system(bash_command)

if RUN_LABEL_PROP:
    bash_command = f'python label_prop_baseline.py {CONFIG_PATH}'
    os.system(bash_command)