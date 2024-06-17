import os
import torch
import numpy as np
import json
from typing import Optional


def load_tokens(file_name: str) -> torch.Tensor:
    tokens = np.load(file_name)
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens


class DataLoader:
    def __init__(self, data_dir: str, batch_size: int, max_seq_len: int, max_steps: Optional[int] = None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_steps = max_steps

        self.shard_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.shard_files.sort()

        self.data_configs = json.load(open(os.path.join(data_dir, 'config.json'), 'r'))

        self.max_possible_steps = self.possible_max_steps()
        if max_steps is None:
            self.max_steps = self.max_possible_steps
        else:
            self.max_steps = min(max_steps, self.max_possible_steps)

    def steps_per_shard(self, token_count: int) -> int:
        if token_count == 0:
            return 0
        steps = token_count // (self.batch_size * self.max_seq_len)
        # For last step we require one extra token to create target
        if (steps * self.batch_size * self.max_seq_len) == token_count:
            steps -= 1
        return steps

    def possible_max_steps(self) -> int:
        token_counts = [shard_config['token_count'] for shard_config in self.data_configs['shards']]
        return sum(map(self.steps_per_shard, token_counts))

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
                raise StopIteration
            self.tokens = load_tokens(self.shard_files[self.current_shard_idx])
            self.currnet_token_idx = 0

        batch_tokens = self.tokens[self.currnet_token_idx:self.currnet_token_idx + token_per_batch + 1]
        x = batch_tokens[:-1].view(self.batch_size, self.max_seq_len)
        y = batch_tokens[1:].view(self.batch_size, self.max_seq_len)
        self.currnet_token_idx += token_per_batch

        return self.steps, x, y


if __name__ == '__main__':
    data_dir = 'pretok_train'
    batch_size = 128
    max_seq_len = 256

    dataloader = DataLoader(data_dir, batch_size, max_seq_len)
    print(f'Max steps: {dataloader.max_steps}')

    for step, x, y in dataloader:
        if step % 100 == 0 or step == dataloader.max_steps - 1:
            print(f'Step {step}: x shape: {x.shape}, y shape: {y.shape}, shard: {dataloader.current_shard_idx}')
