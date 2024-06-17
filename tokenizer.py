import datasets
import os
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import json
import argparse
import warnings

_SEED = 42

def train_tokenizer(datasets: datasets.Dataset, vocab_size: int = 4096, tokenize_model: str = 'tokenizer', train_size_ratio: float = 0.2, min_size: int = 100_000):
    dataset_size = len(datasets)
    min_size = min(min_size, dataset_size)
    train_data_set_size = max(int(len(datasets) * train_size_ratio), min_size)
    print(f"Number of training examples for tokenizer: {train_data_set_size}")

    train_data = datasets.shuffle(seed=_SEED).select(range(train_data_set_size))
    train_text_file = 'tokenizer_train_text.txt'

    # Converting train_data to text file to train sentencepiece tokenizer.
    with open(train_text_file, 'w', encoding='utf-8') as f:
        for data in tqdm(train_data, desc="Writing tokenizer train data"):
            text = data['text'] # Feature name is for TinyStories dataset
            if text:
                f.write(text + '\n')
    
    print(f'Size of tokenizer train data: {os.path.getsize(train_text_file) / 1e6:.2f} MB')

    # Training sentencepiece tokenizer
    spm.SentencePieceTrainer.train(
        input=train_text_file,
        model_prefix=tokenize_model,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True, # Split digits into separate tokens
        allow_whitespace_only_pieces=True, # Allow whitespace only tokens
        byte_fallback=True, # Use byte as fallback
        normalization_rule_name='identity', # No normalization
    )

    print('Tokenizer training completed.')

def pretokenize_datasets_shard(datasets: datasets.Dataset, tokenizer: spm.SentencePieceProcessor, pretok_dir: str, *, shard_size: int, shard_id: int) -> int:
    start, end = shard_id * shard_size, (shard_id + 1) * shard_size
    assert end < len(datasets)

    all_tokens = []
    for data in tqdm(datasets.select(range(start,end)), desc=f"Pretokenizing {pretok_dir} shard {shard_id}"):
        tokens = [tokenizer.bos_id()] + tokenizer.encode(data['text'].strip())
        all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    np.save(os.path.join(pretok_dir, f'shard_{shard_id}.npy'), all_tokens)
    token_count = len(all_tokens)
    print(f'Shard {shard_id} saved to {pretok_dir} with token count {token_count}.')
    return token_count



def pretokenize_datasets(datasets: datasets.Dataset, tokenizer: spm.SentencePieceProcessor, pretok_dir: str, shard_size: int = 20_000, max_shards: int = 50) -> None:
    dataset_size = len(datasets)
    num_shards = min(max_shards, dataset_size // shard_size)
    if num_shards == 0:
        num_shards = 1
        shard_size = dataset_size
        print(f'Number of shards for {pretok_dir} is 0. Setting shard size to {shard_size}.')

    print(f'Number of shards for {pretok_dir}: {num_shards}')

    os.makedirs(pretok_dir, exist_ok=True)
    shard_configs = []
    total_tokens = 0
    for shard_id in range(num_shards):
        token_count = pretokenize_datasets_shard(datasets, tokenizer, pretok_dir, shard_size=shard_size, shard_id=shard_id)
        shard_configs.append({'shard_id': shard_id, 'shard_file': f'shard_{shard_id}.npy', 'token_count': token_count})
        total_tokens += token_count

    config_json = {
        'shard_count': num_shards,
        'shard_size': shard_size,
        'total_tokens': total_tokens,
        'shards': shard_configs
    }
    with open(os.path.join(pretok_dir, 'config.json'), 'w') as f:
        json.dump(config_json, f, indent=4)

    print(f'Pretokenization of {pretok_dir} completed with shard count {num_shards} and total tokens {total_tokens}.')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--output_dir', type=str, default='tokenizer')
    parser.add_argument('--train_tokenizer', action='store_true', default=False)
    parser.add_argument('--pretokenize', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = datasets.load_dataset('roneneldan/TinyStories')

    if args.verbose:
        print("Dataset loaded.")
        print(f"Number of training examples: {len(dataset['train'])}")
        print(f"Number of validation examples: {len(dataset['validation'])}")

    if args.train_tokenizer:
        print("Training tokenizer...")
        tokenizer_model = os.path.join(args.output_dir, 'tokenizer')
        train_tokenizer(dataset['train'], vocab_size=args.vocab_size, tokenize_model=tokenizer_model, train_size_ratio=0.2, min_size=100_00)

        if args.verbose:
            print("Trying to tokenize a sentence")
            # Try tokenizing a sentence
            tokenizer = spm.SentencePieceProcessor(model_file='tokenizer.model')
            sentence = r"Hello, world!\nBye, world!!  ..."
            print(f'Original sentence: {sentence}')
            ids = tokenizer.encode(sentence)
            for id in ids:
                print(f'{id}: {tokenizer.id_to_piece(id)}')
            print(f'Decoded sentence: {tokenizer.decode(ids)}')

    if args.pretokenize:
        print("Pretokenizing datasets...")
        tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(args.output_dir, 'tokenizer.model'))
        # Pretokenize training and validation datasets.
        # For training set, we will tokenize 50 shards of size 20,000 texts each resulting in total 1M texts.
        pretok_train_dir = os.path.join(args.output_dir, 'pretok_train')
        pretok_valid_dir = os.path.join(args.output_dir, 'pretok_valid')
        pretokenize_datasets(dataset['train'], tokenizer, pretok_train_dir, shard_size=20_000, max_shards=50)
        pretokenize_datasets(dataset['validation'], tokenizer, pretok_valid_dir, shard_size=20_000, max_shards=1)

