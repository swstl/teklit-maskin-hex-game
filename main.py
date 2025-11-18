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

for i, board in enumerate(b[0]):
    for pos in board:
        print('x' if pos == -1 else 'o' if pos == 1 else '.', end=' ')
    print()
    print(" "*(i+1), end='')

render_hex_board(b[30])


def render_hex_board(board, game_num=None):
    """
    Render a Hex board with colors in terminal
    
    -1 = Red (Player 1, connects top-bottom)
     1 = Blue (Player 2, connects left-right)
     0 = Empty (white/dot)
    """
    # ANSI color codes
    RED = '\033[91m'
    BLUE = '\033[94m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    size = len(board)
    
    if game_num is not None:
        print(f"\n{BOLD}Game {game_num}{RESET}")
    
    # Top edge label (Red's edge)
    print(f"{RED}{'─' * (size * 2 + 1)}{RESET}")
    
    for i, row in enumerate(board):
        # Left padding for rhombus shape
        print(" " * i, end="")
        
        # Left edge (Blue's edge)
        print(f"{BLUE}\\{RESET} ", end="")
        
        for j, cell in enumerate(row):
            if cell == -1:
                print(f"{RED}⬢{RESET}", end=" ")
            elif cell == 1:
                print(f"{BLUE}⬢{RESET}", end=" ")
            else:
                print(f"{WHITE}·{RESET}", end=" ")
        
        # Right edge (Blue's edge)
        print(f"{BLUE}\\{RESET}")
    
    # Bottom edge label (Red's edge)
    print(" " * size, end="  ")
    print(f"{RED}{'─' * (size * 2 + 1)}{RESET}")
    
    print(f"\n  {RED}⬢ Red (Player 1: Top ↔ Bottom){RESET}")
    print(f"  {BLUE}⬢ Blue (Player 2: Left ↔ Right){RESET}")
