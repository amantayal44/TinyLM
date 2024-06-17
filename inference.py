import torch.nn as nn
import torch
import sentencepiece as spm
import gpt_model
import utils
from tqdm import tqdm
import argparse


def inference(model: gpt_model.GPT, tokenizer: spm.SentencePieceProcessor, text: str, max_len: int = 100, num_samples: int = 1, top_k: int = 10, temp: float = 0.1, device: str = 'cpu') -> str:
    assert max_len <= model.config.block_size, f'max_len should be less than or equal to model block size: {model.config.block_size}'
    assert num_samples > 0, 'num_samples should be greater than 0'
    assert top_k > 0, 'top_k should be greater than 0'
    assert temp >= 0, 'temp should be greater than 0'

    tokens = tokenizer.encode(text)
    inference_steps = max_len - len(tokens)
    if inference_steps <= 0:
        return text

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    tokens = tokens.repeat(num_samples, 1)  # (B, T)

    temp += 1e-6  # for stability

    print(f'Generating text with new {inference_steps} tokens...')
    model.eval()
    for i in tqdm(range(inference_steps), desc='Generating text'):
        with torch.no_grad():
            logits, _ = model(tokens)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)
            logits /= temp
            probs = torch.softmax(logits, dim=-1)  # (B, V)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)  # (B, k)
            # Sample from top k
            next_token_idx = torch.multinomial(top_k_probs, num_samples=1)  # (B, 1)
            # Gather the next token from top k indices
            next_token = torch.gather(top_k_indices, dim=-1, index=next_token_idx)  # (B, 1)
            # Append the next token to the sequence
            tokens = torch.cat((tokens, next_token), dim=-1)  # (B, T+1)

    # Decode the tokens
    generated_text_samples = tokenizer.decode(tokens.tolist())
    for i, text in enumerate(generated_text_samples):
        print(f'Sample {i+1}: {repr(text)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt_model_0.pt', help='Path to the model file')
    parser.add_argument('--config', type=str, default='gpt_config.pt', help='Path to the model config file')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.model', help='Path to the tokenizer model file')
    parser.add_argument('--text', type=str, default='Once upon a time', help='Text to start the generation')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of the generated text')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate')
    parser.add_argument('--top_k', type=int, default=10, help='Top k sampling for generation')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for sampling')
    args = parser.parse_args()

    device = utils.get_device()
    print(f'Using torch device: {device}')

    # Load the model
    config = torch.load(args.config)
    if isinstance(config, gpt_model.GPTConfig):
        model = gpt_model.GPT(config).to(device)
        torch.compile(model)
        model.load_state_dict(torch.load(args.model)) 
        print(f'Model loaded from {args.model} with config {config}')
    else:
        raise ValueError('Invalid model config')

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

    # Generate text
    text = args.text
    inference(model, tokenizer, text, max_len=args.max_len,
              num_samples=args.num_samples, top_k=args.top_k, temp=args.temp, device=device)