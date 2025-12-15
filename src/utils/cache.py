import pandas as pd
import numpy as np
import hashlib
import pickle
import json
import os

def get_graph_cache_path(args):
    """Generate a unique cache filename based on graph parameters"""
    cache_key = {
        'board_size': args.board_size,
        'number_of_boards': args.number_of_boards,
        'hypervector_size': args.hypervector_size,
        'hypervector_bits': args.hypervector_bits,
        'double_hashing': args.double_hashing,
        'one_hot_encoding': args.one_hot_encoding,
    }
    cache_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:8]
 
    os.makedirs('cache', exist_ok=True)
    return f'cache/graphs_{args.board_size}x{args.board_size}_{args.number_of_boards}_{cache_hash}.pkl'

def save_graphs(train_graph, test_graph, symbols, cache_path):
    """Save graphs to disk"""
    print(f"Saving graphs to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train_graph': train_graph,
            'test_graph': test_graph,
            'symbols': symbols
        }, f)
    print("Graphs saved!")

def load_graphs(cache_path):
    """Load graphs from disk"""
    print(f"Loading graphs from cache: {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print("Graphs loaded!")
    return data['train_graph'], data['test_graph'], data['symbols']

def get_or_create_graphs(x_train, x_test, args, create_graphs_fn):
    """Load cached graphs or create new ones"""
    cache_path = get_graph_cache_path(args)
 
    if os.path.exists(cache_path):
        return load_graphs(cache_path)
    else:
        print("No cache found, creating graphs...")
        train_graph, test_graph, symbols = create_graphs_fn(x_train, x_test, args)
        save_graphs(train_graph, test_graph, symbols, cache_path)
        return train_graph, test_graph, symbols









def load_hex_data_cached(csv_path):
    """Load the Hex game CSV data with caching for speed"""
    # Check for cached numpy version
    cache_path = csv_path.replace('.csv', '_cached.npz')
 
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return list(cached['boards']), cached['labels']
 
    print(f"No cache found, loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
 
    board_columns = [col for col in df.columns if col.startswith('cell')]
 
    max_row = max(int(col.replace('cell', '').split('_')[0]) for col in board_columns)
    max_col = max(int(col.replace('cell', '').split('_')[1]) for col in board_columns)
    board_size = max(max_row, max_col) + 1
 
    # Vectorized approach - much faster than iterrows
    boards = []
    board_data = df[board_columns].values
 
    for row_data in board_data:
        board = np.zeros((board_size, board_size), dtype=int)
        for idx, col in enumerate(board_columns):
            parts = col.replace('cell', '').split('_')
            r, c = int(parts[0]), int(parts[1])
            board[r, c] = row_data[idx]
        boards.append(board)
 
    labels = np.array(df['winner'].values)
 
    # Save to cache
    print(f"Saving to cache: {cache_path}")
    np.savez_compressed(cache_path, boards=np.array(boards, dtype=object), labels=labels)
 
    return boards, labels
