from src.render import draw_simple_board

import numpy as np
import pandas as pd
import argparse


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10, type=int)
    parser.add_argument("--T", default=100, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=32, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=10000, type=int)
    parser.add_argument("--max-included-literals", default=4, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


def load_hex_data(csv_path):
    """Load the Hex game CSV data"""
    df = pd.read_csv(csv_path)
 
    board_columns = [col for col in df.columns if col.startswith('cell')]
 
    max_row = max(int(col.replace('cell', '').split('_')[0]) for col in board_columns)
    max_col = max(int(col.replace('cell', '').split('_')[1]) for col in board_columns)
    board_size = max(max_row, max_col) + 1  # +1 because indices start at 0
 
    boards = []
    for _, row in df.iterrows():
        board = np.zeros((board_size, board_size), dtype=int)
        for col in board_columns:
            parts = col.replace('cell', '').split('_')
            r, c = int(parts[0]), int(parts[1])
            board[r, c] = row[col]
        boards.append(board)
 
    labels = df['winner'].values
    return boards, labels


b, t = load_hex_data('data/hex_games_1000_size_2.csv')
draw_simple_board(b[0], radius=100, red_val=-1, blue_val=1)
int(t[2])
b[0]

