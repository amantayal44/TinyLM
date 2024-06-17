from functools import partial
import torch
import dataloader
import gpt_model
from tqdm import tqdm
import time
import argparse
import os
import utils
import matplotlib.pyplot as plt
import wandb
import math
import json


def val_loss(model, data_loader):
    model.eval()
    total_loss = 0
    for _, x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item()
    total_loss /= data_loader.max_steps
    return total_loss

class CosineScheduler:
    def __init__(self, optimizer, lr_max, lr_min, warmup_steps, total_steps):
        print(f'CosineScheduler created with lr_max: {lr_max}, lr_min: {lr_min}, warmup_steps: {warmup_steps}, total_steps: {total_steps}')
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
    
    def __call__(self, step):
        if step < self.warmup_steps:
            lr = self.lr_max * (step + 1) / self.warmup_steps
        elif step >= self.total_steps:
            lr =  self.lr_min
        else:
            decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.lr_min + (self.lr_max - self.lr_min) * coeff

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr



def plot_graph(train_losses, val_losses, output_dir):
    plt.clf()
    plt.plot([x['step'] for x in train_losses], [x['loss'] for x in train_losses], label='Train Loss')
    plt.plot([x['step'] for x in val_losses], [x['loss'] for x in val_losses], label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for model, config and logs')
    parser.add_argument('--tokenizer_dir', type=str, help='Directory containing tokenizer model and pretokenized data')
    parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size for the model')
    parser.add_argument('--block_size', type=int, default=256, help='Block size for the model')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--n_dim', type=int, default=256, help='Embedding dimension for the model')
    parser.add_argument('--max_train_steps', type=int, default=-1, help='Maximum training steps, set to -1 for full training')
    parser.add_argument('--max_val_steps', type=int, default=-1, help='Maximum validation steps, set to -1 for full validation. For validation batch size is doubled.')
    parser.add_argument('--val_loss_steps', type=int, default=10, help='Calculate validation loss after every val_loss_steps steps')
    parser.add_argument('--save_steps', type=int, default=1, help='Save model after every save_steps validation steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Max learning rate for training')
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb for logging')
    args = parser.parse_args()

    device = 'cpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Using torch device: {device}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data loader
    train_loader = dataloader.DataLoader(f'{args.tokenizer_dir}/pretok_train', batch_size=args.batch_size, max_seq_len=args.block_size, max_steps=args.max_train_steps if args.max_train_steps != -1 else None)
    val_loader = dataloader.DataLoader(f'{args.tokenizer_dir}/pretok_valid', batch_size=args.batch_size * 2, max_seq_len=args.block_size, max_steps=args.max_val_steps if args.max_val_steps != -1 else None)
    max_train_steps = train_loader.max_steps
    print(f'Train data loaded with max steps: {max_train_steps}')
    print(f'Val data loaded with max steps: {val_loader.max_steps}')

    # Create model
    # TODO: Add support for other model types
    model_config = gpt_model.GPTConfig(vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_dim=args.n_dim)
    torch.save(model_config, os.path.join(args.output_dir, 'model_config.pt'))
    model = gpt_model.GPT(model_config).to(device)
    torch.compile(model)
    print(f'Model created with configs: {model_config}')
    print(f'Model parameters count: {utils.count_parameters(model)}')

    # Create optimizer
    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=0.1)
    warmup_steps = min(500, max_train_steps // 10)
    scheduler = CosineScheduler(optimizer, lr_max=args.lr, lr_min=0.1*args.lr, warmup_steps=warmup_steps, total_steps=max_train_steps)
    print(f'Optimizer created with lr: {args.lr}, warmup_steps: {warmup_steps}')

    # Data for plotting graphs
    train_losses = []
    val_losses = []
    
    chkpt_list = []

    best_saved_model, best_saved_model_loss = None, 10000

    if args.wandb:
     wandb.init(project='TinyLM')

    print(f'Starting training... for {max_train_steps} steps.')
    total_start = time.time()
    for step, x, y in train_loader:

        # Training step
        start = time.time()
        model.train()
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = scheduler(step)
        optimizer.step()
        print(f'Step {step + 1}/{max_train_steps}, Loss: {loss.item():.6f}, Norm: {norm:.4f}, Learning Rate: {lr:.4e} Time taken: {time.time() - start:.2f} sec, Total time: {time.time() - total_start:.2f} sec, ETA: {(time.time() - total_start) * (max_train_steps - step) / (step + 1):.2f} sec')
        train_losses.append({'step': step, 'loss': loss.item(), 'norm': norm, 'lr': lr})
        if args.wandb:
            wandb.log({'loss/train': loss.item(), 'step': step, 'norm': norm})

        # Validation step
        if step % args.val_loss_steps == 0 or step == max_train_steps - 1:
            start = time.time()
            val_loss_ = val_loss(model, val_loader)
            print(f'===> Step {step+1}, Val Loss: {val_loss_:.4f}, Time taken: {time.time() - start:.4f} sec')
            val_losses.append({'step': step, 'loss': val_loss_})
            if args.wandb:
                wandb.log({'loss/val': val_loss_, 'step': step})

            if (step // args.val_loss_steps + 1) % args.save_steps == 0 or step == max_train_steps - 1:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_state_{step}.pt'))
                chkpt_list.append({'model': f'model_state_{step}.pt', 'loss': val_loss_})
                print(f'===> Model saved at step {step+1}')
                if val_loss_ < best_saved_model_loss:
                    best_saved_model_loss = val_loss_
                    best_saved_model = f'model_state_{step}.pt'
            
        
        if step % 10 == 0:
            # Save new plot after every validation step
            plot_graph(train_losses, val_losses, args.output_dir)
            print(f'===> Updated loss plot saved at step {step+1}')

            all_losses = {'train': train_losses, 'val': val_losses}
            torch.save(all_losses, os.path.join(args.output_dir, 'all_losses.pt'))


    print(f'Training completed. Best model saved at {best_saved_model} with loss: {best_saved_model_loss:.4f}')

    # Store details of the training run
    details = {
        'best_model': {
            'model': best_saved_model,
            'loss': best_saved_model_loss
        },
        'checkpoints': chkpt_list,
    }
    with open(os.path.join(args.output_dir, 'details.json'), 'w') as f:
        json.dump(details, f, indent=4)

    if args.wandb:
        wandb.finish()
