import os

import torch
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule

from model import CTRNNWithHead
from trainer import Trainer

_hp = {
    'seq_len': 100,
    'batch_size': 64,
    'steps': 10000,
    'hidden_size': 256,
    'nonlinearlity': 'relu',
    'noise_std': 0.02,
    'dt': 20,
    'dim_ring': 32,
}
ckpt_path = './ckpt'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    os.makedirs(ckpt_path, exist_ok=True) # To save model checkpoints

    ## Make simulation dataset
    tasks = ngym.get_collection('yang19')
    env = ScheduleEnvs(
        envs=[gym.make(task, dt=_hp['dt'], dim_ring=_hp['dim_ring']) for task in tasks],
        schedule=RandomSchedule(len(tasks)),
        env_input=True
    )
    dataset = ngym.Dataset(
        env,
        batch_size=_hp['batch_size'],
        seq_len=_hp['seq_len']
    )
    ob_size = dataset.env.observation_space.shape[0]
    act_size = env.action_space.n

    ## Model
    model = CTRNNWithHead(
        input_size=ob_size,
        hidden_size=_hp['hidden_size'],
        output_size=act_size,
        dt=_hp['dt'],
        nonlinearlity=_hp['nonlinearlity'],
        noise_std=_hp['noise_std'],
        device=device,
    )

    ## Train
    trainer = Trainer(model, dataset, ckpt_path, device)
    trainer.train(steps=_hp['steps'])


if __name__ == "__main__":
    main()