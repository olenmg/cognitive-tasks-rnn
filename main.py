import os

import torch
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule

from dataset import load_dataset
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
    env, dataset = load_dataset(_hp)
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
    ).to(device)

    ## Train
    trainer = Trainer(model, dataset, ckpt_path, device)
    trainer.train(steps=_hp['steps'])


if __name__ == "__main__":
    main()
