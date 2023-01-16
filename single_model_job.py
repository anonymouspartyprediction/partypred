import wandb
from simple_slurm import Slurm
from omegaconf import OmegaConf
import sys

sweep_conf_path = sys.argv[1]
num_jobs = sys.argv[2]

slurm = Slurm(
    array=range(0,int(num_jobs)),
    cpus_per_task=4,
    job_name='sweep',
    gres='gpu:rtx8000:1',
    mem='48gb',
    output=f'logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
)

sweep_conf = OmegaConf.to_container(OmegaConf.load(sweep_conf_path))

ent = sweep_conf['entity']
proj = sweep_conf['project']

sweep_id = wandb.sweep(sweep_conf, project=proj, entity=ent)

slurm.sbatch(f'wandb agent --count 1 {ent}/{proj}/{sweep_id}')