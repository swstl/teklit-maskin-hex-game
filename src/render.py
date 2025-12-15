import tkinter as tk
import math
import numpy as np
import pandas as pd
import os

def create_hexagon(canvas, cx, cy, radius, **kwargs):
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.extend([x, y])
    return canvas.create_polygon(points, **kwargs)

def create_border(canvas, direction, cx, cy, radius, length, **kwargs):
    hex_width = radius * math.sqrt(3)

    if direction == 'down':
        for _ in range(length):
            points = []
            for j in range (3, 6):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.extend([x, y])
            for j in range (6, 2, -1):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy - 10 + radius * math.sin(angle)
                points.extend([x, y])
            canvas.create_polygon(points, **kwargs)
            cx += hex_width 

    elif direction == 'up':
        for _ in range(length):
            points = []
            for j in range (3):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.extend([x, y])
            for j in range (3, -1, -1):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy + 10 + radius * math.sin(angle)
                points.extend([x, y])
            canvas.create_polygon(points, **kwargs)
            cx += hex_width 

    elif direction == 'right':
        for _ in range(length):
            points = []
            for j in range(1, 4):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.extend([x, y])
            for j in range (3, 0, -1):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx - 10 + radius * math.cos(angle)
                y = cy + 5 + radius * math.sin(angle)
                points.extend([x, y])
            cy += radius * 1.5
            cx += hex_width / 2 
            canvas.create_polygon(points, **kwargs)

    elif direction == 'left':
        for _ in range(length):
            points = []
            for j in range(3):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.extend([x, y])
            for j in range (3, 0, -1):
                angle = math.pi / 6 + j * math.pi / 3
                x = cx - 10 + radius * math.cos(angle)
                y = cy + 5 + radius * math.sin(angle)
                points.extend([x, y])
            cy += radius * 1.5
            cx += hex_width / 2 
            canvas.create_polygon(points, **kwargs)


def draw_simple_board(board, radius, red_val, blue_val):
    hex_width = radius * math.sqrt(3)
    hex_height = radius * 1.5

    root = tk.Tk()
    canvas = tk.Canvas(root,
                       width=hex_width*len(board[0])+hex_width/2*len(board),
                       height=hex_height*(len(board)+1)
                       )
    canvas.pack()

    create_border(
        canvas, 
        direction='down', 
        cx=hex_width, 
        cy=hex_height, 
        radius=radius, 
        length=len(board[0]), 
        fill="lightblue", 
        outline="black")

    create_border(
        canvas, 
        direction='right', 
        cx=hex_width, 
        cy=hex_height, 
        radius=radius, 
        length=len(board[0]), 
        fill="pink", 
        outline="black")

    for i, row in enumerate(board):
        for j, col in enumerate(row):
            color = "red" if col == red_val else "blue" if col == blue_val else "lightgray"
            create_hexagon(
                canvas, 
                cx=hex_width*(i/2+1+j), 
                cy=hex_height*(1+i), 
                radius=radius, 
                fill=color, 
                outline="black")

    root.mainloop()


def simple_hex_render(b):
    for i, board in enumerate(b[0]):
        for pos in board:
            print('x' if pos == -1 else 'o' if pos == 1 else '.', end=' ')
        print()
        print(" "*(i+1), end='')



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

if __name__ == "__main__":    # Example usage
    boards, labels = _load_hex_data('src/dataset/hex_games_10_size_10_stop_3.csv')
    draw_simple_board(boards[0], radius=30, red_val=-1, blue_val=1)#         from src.generate_hex_data import generate_hex_data
