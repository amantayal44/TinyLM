from functools import partial
import torch
import dataloader
import gpt_model
import llama_model
from tqdm import tqdm
import time
import argparse
import os
import utils
import matplotlib.pyplot as plt
import wandb
import math
import json
from typing import Union

def load_checkpoint(checkpoint_path, device) -> Union[gpt_model.GPT, llama_model.Llama]:
    model_dict = torch.load(checkpoint_path)
    model_type = model_dict['type']

    if model_type == 'GPT':
        model = gpt_model.GPT(model_dict['config'])
    elif model_type == 'Llama':
        model = llama_model.Llama(model_dict['config'])
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    model = model.to(device)
    model.load_state_dict(model_dict['state'])
    print(f'Model loaded from checkpoint {checkpoint_path}')

    return model

def cread_model_from_args(args, device) -> Union[gpt_model.GPT, llama_model.Llama]:
    if args.model_type == 'GPT':
        config = gpt_model.GPTConfig(vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_dim=args.n_dim, n_head=args.n_head)
        model = gpt_model.GPT(config)
    elif args.model_type == 'Llama':
        config = llama_model.LlamaConfig(vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_dim=args.n_dim, n_head=args.n_head, n_kv_head=args.n_kv_head)
        model = llama_model.Llama(config)
    else:
        raise ValueError(f'Unsupported model type: {args.model_type}')

    model = model.to(device)
    return model

@torch.no_grad()
def val_loss(model: gpt_model.GPT, data_loader: dataloader.DataLoader):
    model.eval()
    total_loss = 0
    for _, x, y in data_loader:
        _, loss = model(x, y)
        total_loss += loss.item()
    total_loss /= data_loader.max_steps
    return total_loss

class CosineScheduler:
    def __init__(self, optimizer, lr_max, lr_min, warmup_steps, total_steps):
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
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint to be used for training model, No checkpoint is used if set to "".')
    parser.add_argument('--model_type', type=str, default='GPT', help='Type of model, only supports "GPT" and "Llama"', choices=['GPT', 'Llama'])
    parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size for the model')
    parser.add_argument('--block_size', type=int, default=256, help='Block size for the model')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--n_dim', type=int, default=256, help='Embedding dimension for the model')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads in the model')
    parser.add_argument('--n_kv_head', type=int, default=8, help='Number of key-value attention heads in the model, Only for Lllama model.')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of steps to accumulate gradients over')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training, if grad_accum_steps > 1, this is micro batch size')
    parser.add_argument('--max_train_steps', type=int, default=-1, help='Maximum training steps, set to -1 for 1 epoch, if grad_accum_steps > 1, this is steps per micro batch')
    parser.add_argument('--max_val_steps', type=int, default=-1, help='Maximum validation steps, set to -1 for 1 epoch. For validation batch size is doubled.')
    parser.add_argument('--val_loss_steps', type=int, default=10, help='Calculate validation loss after every val_loss_steps grad accum steps')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max learning rate for training')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Min learning rate for training')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb for logging')
    args = parser.parse_args()

    device = utils.get_device()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate total training steps
    train_steps_per_epoch = dataloader.possible_max_steps_in_dir(f'{args.tokenizer_dir}/pretok_train', args.batch_size, args.block_size)
    max_train_steps = args.max_train_steps if args.max_train_steps != -1 else train_steps_per_epoch
    max_grad_steps = max_train_steps // args.grad_accum_steps
    max_train_steps = max_grad_steps * args.grad_accum_steps

    # Create data loaders
    train_loader = dataloader.DataLoader(f'{args.tokenizer_dir}/pretok_train', batch_size=args.batch_size, max_seq_len=args.block_size, device=device, max_steps=max_train_steps)
    val_loader = dataloader.DataLoader(f'{args.tokenizer_dir}/pretok_valid', batch_size=args.batch_size * 2, max_seq_len=args.block_size, device=device, max_steps=args.max_val_steps * args.grad_accum_steps if args.max_val_steps != -1 else None)
    max_train_steps = train_loader.max_steps
    print(f'Train data loaded with max steps: {max_train_steps}')
    print(f'Val data loaded with max steps: {val_loader.max_steps}')

    # Create model
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, device)
    else:
        model = cread_model_from_args(args, device)

    torch.compile(model)
    print(f'Model created with configs: {model.config}')
    print(f'Model parameters count: {utils.count_parameters(model)}')

    # Create optimizer and scheduler
    optimizer = model.configure_optimizers(lr=args.min_lr, weight_decay=0.1)
    warmup_steps = min(args.warmup_steps, max_grad_steps // 10) # minumum of warmput_steps steps or 10% of total grad steps.
    scheduler = CosineScheduler(optimizer, lr_max=args.max_lr, lr_min=args.min_lr, warmup_steps=warmup_steps, total_steps=max_grad_steps)
    print(f'Optimizer created with scheduler: cosine(warmup steps: {warmup_steps}, total steps: {max_grad_steps}, max lr: {args.max_lr}, min lr: {args.min_lr}) and weight decay: 0.1')

    # Data for plotting graphs
    train_losses = []
    val_losses = []
    
    # Checkpointing
    best_saved_model_step, best_saved_model_loss = -1, 10000

    if args.wandb:
     wandb.init(project='TinyLM')

    print(f'Starting training... for {max_grad_steps} steps.')

    overall_T = time.time()
    step_T = time.time()
    plot_T = time.time()
    val_step = 0
    
    train_loader_iter = iter(train_loader)
    micro_step, x, y = next(train_loader_iter)

    while micro_step < train_loader.max_steps:
        curr_micro_step = micro_step

        # Training step
        model.train()
        _, loss = model(x, y)
        loss = loss / args.grad_accum_steps # Normalize the loss for gradient accumulation
        
        # Immediate load next batch asynchrounously
        if curr_micro_step + 1 < train_loader.max_steps:
            micro_step, x, y = next(train_loader_iter)
        else:
            micro_step = train_loader.max_steps
        
        loss.backward()

        # Gradient accumulation
        if (curr_micro_step + 1) % args.grad_accum_steps == 0:
            grad_step = curr_micro_step // args.grad_accum_steps

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = scheduler(grad_step)
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item() * args.grad_accum_steps
            overall_D, step_D = time.time() - overall_T, time.time() - step_T
            print(f'Step {grad_step}/{max_grad_steps}, Loss: {loss_value:.6f}, Norm: {norm:.4f}, Learning Rate: {lr:.4e} Time taken: {step_D:.2f} sec, Total time: {overall_D:.2f} sec')
            train_losses.append({'step': grad_step, 'loss': loss_value, 'norm': norm, 'lr': lr, 'time': step_D})
            if args.wandb:
                wandb.log({'loss/train': loss_value, 'step': grad_step, 'norm': norm, 'lr': lr, 'time': step_D})

            # Calculate validation loss
            if args.val_loss_steps > 0 and (grad_step % args.val_loss_steps == 0 or grad_step == max_grad_steps - 1):
                val_T = time.time()
                val_loss_ = val_loss(model, val_loader)
                val_D = time.time() - val_T

                # Logging validation loss
                print(f'===> Validation Loss: {val_loss_:.6f}, Time taken: {val_D:.4f} sec')
                val_losses.append({'step': grad_step, 'loss': val_loss_, 'time': val_D})
                if args.wandb:
                    wandb.log({'loss/val': val_loss_, 'step': grad_step})
                
                model_dict = {'type': args.model_type, 'config': model.config, 'state': model.state_dict()}

                # Save latest model
                torch.save(model_dict, os.path.join(args.output_dir, f'last_model.pt'))

                # Save best model
                if val_loss_ < best_saved_model_loss:
                    best_saved_model_loss = val_loss_
                    best_saved_model_step = grad_step
                    torch.save(model_dict, os.path.join(args.output_dir, f'best_model.pt'))
                    print(f'===> Best model saved at step {grad_step} with loss: {val_loss_:.6f}')
            
            # Update the plot and save losses after every 15s.
            if time.time() - plot_T > 15 or grad_step == max_grad_steps - 1:
                plot_graph(train_losses, val_losses, args.output_dir)
                all_losses = {'train': train_losses, 'val': val_losses}
                torch.save(all_losses, os.path.join(args.output_dir, 'all_losses.pt'))
                plot_T = time.time()

            # Reset step timer
            step_T = time.time()


    print(f'Training completed. Best model saved at {best_saved_model_step} with loss: {best_saved_model_loss:.4f}')

    # Store details of the training run
    details = {
        'best_model': {
            'model_step': best_saved_model_step,
            'loss': best_saved_model_loss
        },
        'validation loss': [{'step': val_loss['step'], 'loss': val_loss['loss']} for val_loss in val_losses]
    }
    with open(os.path.join(args.output_dir, 'details.json'), 'w') as f:
        json.dump(details, f, indent=4)

    if args.wandb:
        wandb.finish()
