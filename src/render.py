import tkinter as tk
import math

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

