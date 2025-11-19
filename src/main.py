# import kagglehub
# import shutil
# import os
# from GraphTsetlinMachine.graphs import Graphs
# from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
#
#
#
# # Download latest version
# path = kagglehub.dataset_download("cholling/game-of-hex")
# data_path = os.path.join('data')
# shutil.move(path, data_path)



import numpy as np
import pandas as pd

from src.render import draw_simple_board

def load_hex_data(csv_path):
    """Load the Hex game CSV data"""
    df = pd.read_csv(csv_path)

    # Get board columns
    board_columns = [col for col in df.columns if col.startswith('cell')]

    # Reshape each row into 7x7 board
    boards = []
    for _, row in df.iterrows():
        board = np.zeros((7, 7), dtype=int)
        for col in board_columns:
            # Parse cell position: cell0_0 -> row 0, col 0
            parts = col.replace('cell', '').split('_')
            r, c = int(parts[0]), int(parts[1])
            board[r, c] = row[col]
        boards.append(board)

    labels = df['winner'].values

    return boards, labels






b, t = load_hex_data('data/hex_games_1_000_size_7.csv')
draw_simple_board(b[2], 20, red_val=-1, blue_val=1)
int(t[2])
