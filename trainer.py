import os

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


class Trainer:
    def __init__(self, model, dataset, ckpt_path, device):
        self.model = model
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def eval(self, num_trials=1000):
        env = self.dataset.env

        acc = 0.
        for _ in range(num_trials):
            env.new_trial()
            inputs, labels = env.ob, env.gt
            inputs = torch.from_numpy(inputs).float().to(self.device)
            inputs.unsqueeze(1) # batch axis (batch size as 1)

            pred = self.model(inputs)
            pred = torch.argmax(pred, axis=-1).cpu().numpy()
            acc += pred[-1, 0] == labels[-1]
        acc /= num_trials
        return acc 

    def train_step(self):
        inputs, labels = self.dataset()
        inputs = torch.from_numpy(inputs).float().to(self.device)
        labels = torch.from_numpy(labels).long().to(self.device)

        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.flatten())
        loss.backward()
        # clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def train(self, steps):
        print_step = steps // 20

        best_acc = float('-inf')
        total_loss = 0.
        for step in tqdm(range(steps)):
            loss = self.train_step()
            total_loss += loss
            if (step + 1) % print_step == 0:
                acc = self.eval()
                print(f"[STEP {step}/{steps}] Loss: {total_loss / steps: .3f} | Eval Acc.: {acc: .3f}")

                if acc > best_acc:
                    print(f"Best model has been saved.")
                    best_acc = acc
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.ckpt_path, f'step{step}_{acc: .3f}.pt')
                    )
