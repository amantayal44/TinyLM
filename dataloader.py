import os
import torch
import numpy as np
import json
from typing import Optional, Any

def load_tokens(file_name: str) -> torch.Tensor:
    tokens = np.load(file_name)
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens

def steps_per_shard(token_count: int, batch_size: int, max_seq_len: int) -> int:
    if token_count == 0:
        return 0
    steps = token_count // (batch_size * max_seq_len)
    # For last step we require one extra token to create target
    if (steps * batch_size * max_seq_len) == token_count:
        steps -= 1
    return steps

def possible_max_steps(config_json: dict[str, Any], batch_size: int, max_seq_len: int) -> int:
    token_counts = [shard_config['token_count'] for shard_config in config_json['shards']]
    return sum(map(lambda x: steps_per_shard(x, batch_size, max_seq_len), token_counts))

def possible_max_steps_in_dir(data_dir: str, batch_size: int, max_seq_len: int) -> int:
    config_json = json.load(open(os.path.join(data_dir, 'config.json'), 'r'))
    return possible_max_steps(config_json, batch_size, max_seq_len)

class DataLoader:
    def __init__(self, data_dir: str, batch_size: int, max_seq_len: int, max_steps: Optional[int] = None, device: Optional[str] = None):
        """
        DataLoader for TinyLM dataset

        Args:
            data_dir (str): Path to the data directory
            batch_size (int): Batch size
            max_seq_len (int): Maximum sequence length
            max_steps (int, optional): Maximum number of steps to iterate. Defaults to None. If None, iterate over 1 epoch.
            device (str, optional): Device to move the tensors. Defaults to None.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_steps = max_steps
        self.device = device

        self.shard_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.shard_files.sort()

        self.data_configs = json.load(open(os.path.join(data_dir, 'config.json'), 'r'))

        self.max_possible_steps = self.possible_max_steps()
        if max_steps is None:
            self.max_steps = self.max_possible_steps

    def possible_max_steps(self) -> int:
        return possible_max_steps(self.data_configs, self.batch_size, self.max_seq_len)

    def __iter__(self) -> 'DataLoader':
        self.current_shard_idx = 0
        self.currnet_token_idx = 0
        self.tokens = load_tokens(self.shard_files[self.current_shard_idx])
        self.steps = -1
        return self

    def __next__(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        self.steps += 1
        if self.steps >= self.max_steps:
            raise StopIteration

        token_per_batch = self.batch_size * self.max_seq_len

        # Check if current shard has enough tokens for next batch
        if self.currnet_token_idx + token_per_batch + 1 > len(self.tokens):
            self.current_shard_idx += 1
            if self.current_shard_idx >= len(self.shard_files):
                self.current_shard_idx = 0

            self.tokens = load_tokens(self.shard_files[self.current_shard_idx])
            self.currnet_token_idx = 0

        batch_tokens = self.tokens[self.currnet_token_idx:self.currnet_token_idx + token_per_batch + 1]
        x = batch_tokens[:-1].view(self.batch_size, self.max_seq_len)
        y = batch_tokens[1:].view(self.batch_size, self.max_seq_len)
        self.currnet_token_idx += token_per_batch

        if self.device:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return self.steps, x, y


if __name__ == '__main__':
    data_dir = 'pretok_train'
    batch_size = 512
    max_seq_len = 512
    max_steps = 1000 # Use None for 1 full epoch.

    dataloader = DataLoader(data_dir, batch_size, max_seq_len, max_steps=max_steps)
    print(f'Max steps: {dataloader.max_steps}')

    for step, x, y in dataloader:
        if step % 100 == 0 or step == dataloader.max_steps - 1:
            print(f'Step {step}: x shape: {x.shape}, y shape: {y.shape}, shard: {dataloader.current_shard_idx}')
