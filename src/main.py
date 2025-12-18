from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
# from utils.cache import get_or_create_graphs
from graphs.v2 import create_graphs

import pandas as pd
import numpy as np
import subprocess
import argparse
import shutil
import time
import os

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves-before-win", "-m", default=5, type=int)
    parser.add_argument("--number-of-boards", "-n", default=100000, type=int)
    parser.add_argument("--board-size", "-b", default=19, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--number-of-clauses", default=32000, type=int)
    parser.add_argument("--T", default=6000, type=int)
    parser.add_argument("--s", default=1.5, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=256, type=int)
    parser.add_argument("--hypervector-bits", default=4, type=int)
    parser.add_argument("--message-size", default=82, type=int)
    parser.add_argument("--message-bits", default=4, type=int)
    parser.add_argument("--double-hashing", dest="double_hashing", default=False, action="store_true")
    parser.add_argument("--one-hot-encoding", dest="one_hot_encoding", default=False, action="store_true")
    parser.add_argument("--max-included-literals", default=250, type=int) #makes the clauses include more specific things about the dataset (literals), too much can cause overfitting??

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def print_args(args):
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

def _load_hex_data(csv_path):
    """Load the Hex game CSV data with caching"""
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
 
    print(f"Saving to cache: {cache_path}")
    np.savez_compressed(cache_path, boards=np.array(boards, dtype=object), labels=labels)
 
    return boards, labels



##################################################
################  Split the data  ################
##################################################
def get_hex_games(split, args, folder='data'):
    os.makedirs(folder, exist_ok=True)

    how_many = args.number_of_boards
    board_size = args.board_size
    moves_before_win = args.moves_before_win

    if moves_before_win > 0:
        csv_file = f'hex_games_{how_many}_size_{board_size}_stop_{moves_before_win}.csv'
    else:
        csv_file = f'hex_games_{how_many}_size_{board_size}.csv'

    try:
        b, t = _load_hex_data(os.path.join(folder, csv_file))

        print(f"Loaded {len(b)} games from {csv_file}")

        split = int(len(b)*split)
        indices = np.random.permutation(len(b))

        X_train = [b[i] for i in indices[:split]]
        X_test = [b[i] for i in indices[split:]]
        Y_train = np.array([0 if y == -1 else y for y in t[indices[:split]]], dtype=np.uint32)
        Y_test = np.array([0 if y == -1 else y for y in t[indices[split:]]], dtype=np.uint32)

        return X_train, Y_train, X_test, Y_test

    except FileNotFoundError:
        # generate the file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hex_executable = os.path.join(script_dir, 'dataset', 'hex')
        subprocess.run([hex_executable, '-b', str(board_size), '-g', str(how_many), '-s', str(moves_before_win)], check=True)
        # move the file
        shutil.move(csv_file, os.path.join(folder, csv_file))
        # rerun
        return get_hex_games(split, args)




##################################################
################   Train the tm   ################
##################################################
def train_tm(tm, graph_train, Y_train, graph_test, Y_test, args):
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        before = time.time()
        tm.fit(graph_train, Y_train, epochs=1, incremental=True)
        epoch_time = time.time() - before

        pred_test = tm.predict(graph_test)
        acc_test = 100 * np.mean(pred_test == Y_test)

        pred_train = tm.predict(graph_train)
        acc_train = 100 * np.mean(pred_train == Y_train)

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"Train Accuracy: {acc_train:.2f}% "
            f"Test Accuracy: {acc_test:.2f}% "
            f"Time: {epoch_time:.2f}s", flush=True
        )



###################################################
################  Actual  running  ################
###################################################
# 1. get dataset : get_hex_games
# 2. Create the graphs : create_graphs
# 3. Train the tm : train_tm

args = default_args()

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    depth=args.depth,
    max_included_literals=args.max_included_literals,
    number_of_state_bits=args.number_of_state_bits,
    one_hot_encoding=args.one_hot_encoding,
    double_hashing=args.double_hashing,
    message_size=args.message_size,
    message_bits=args.message_bits,
    grid=(16*13, 1, 1),
    block=(128, 1, 1)
)

print_args(args)
x_train, y_train, x_test, y_test = get_hex_games(0.8, args)
# train_graph, test_graph, symbols = get_or_create_graphs(x_train, x_test, args, create_graphs)
train_graph, test_graph, symbols = create_graphs(x_train, x_test, args)
train_tm(tm, train_graph, y_train, test_graph, y_test, args)
