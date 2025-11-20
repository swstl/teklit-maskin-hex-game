from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

import pandas as pd
import numpy as np
import subprocess
import argparse
import shutil
import time
import os

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", "-d", default=10000, type=int)
    parser.add_argument("--board_size", "-b", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--number-of-clauses", default=5000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=256, type=int)
    parser.add_argument("--hypervector-bits", default=4, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=4, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)

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
 
    labels = np.array(df['winner'].values)
    return boards, labels


##################################################
################  Split the data  ################
##################################################
#TODO: make this generate if file not found
def get_hex_games(split, how_many=1000, board_size=3):
    csv_path = f'data/hex_games_{how_many}_size_{board_size}.csv'
    try:
        b, t = _load_hex_data(csv_path)

        print(f"Loaded {len(b)} games from {csv_path}")

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
        subprocess.run([hex_executable, '-b', str(board_size), '-g', str(how_many)], check=True)

        # move the file
        os.makedirs('data', exist_ok=True)
        generated_file = f'hex_games_{how_many}_size_{board_size}.csv'
        shutil.move(generated_file, csv_path)

        # rerun
        return get_hex_games(split, how_many, board_size)



###################################################
################ Create the graphs ################
###################################################

def create_graphs(X_train, X_test, args):
    print(f"Creating graphs on dataset of length {len(X_train)}, and {len(X_test)}...")
    def get_neighbours(row, col, bord_size):
        neighbours = []
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < bord_size and 0 <= nc < bord_size:
                neighbours.append((nr, nc))
        return neighbours

    def configure_graphs(graph, X):
        bord_size = len(X[0])
        num_graphs = len(X)
        # sets the number of nodes for each graph (and their id)
        cells = bord_size * bord_size
        for id in range(num_graphs):
            graph.set_number_of_graph_nodes(id, cells)

        # adds the number of nodes for each graph with how many edges they expect
        graph.prepare_node_configuration()
        for id in range(num_graphs):
            for row in range(bord_size):
                for col in range(bord_size):
                    cell_name = f'cell_{row}_{col}' # just the node identity
                    num_edges = len(get_neighbours(row, col, bord_size))
                    graph.add_graph_node(id, cell_name, num_edges)

        # add the edges to te previous added nodes, in the graphs
        graph.prepare_edge_configuration()
        for id in range(num_graphs):
            board = X[id]
            for row in range(bord_size):
                for col in range(bord_size):
                    cell_name = f'cell_{row}_{col}'

                    # add neighbours:
                    neighbours = get_neighbours(row, col, bord_size)

                    for nr, nc in neighbours:
                        n_name = f'cell_{nr}_{nc}'
                        edge_type = 'neighbour'
                        graph.add_graph_node_edge(id, cell_name, n_name, edge_type)

                    # add the color of each node:
                    cell_value = board[row, col]
                    match cell_value:
                        case -1:
                            graph.add_graph_node_property(id, cell_name, 'red')
                        case 0:
                            graph.add_graph_node_property(id, cell_name, 'empty')
                        case 1:
                            graph.add_graph_node_property(id, cell_name, 'blue')

                    # position of each node:
                    graph.add_graph_node_property(id, cell_name, f'row_{row}')
                    graph.add_graph_node_property(id, cell_name, f'col_{col}')

                    # TODO: might just add one of the propeties to the node (either border node or corner node)
                    # then the matches can be combined #
                    # special markers (boarders)
                    match (row, col):
                        case (0, 0):
                            graph.add_graph_node_property(id, cell_name, 'top')
                            graph.add_graph_node_property(id, cell_name, 'left')
                        case (0, c) if c == bord_size - 1:
                            graph.add_graph_node_property(id, cell_name, 'top')
                            graph.add_graph_node_property(id, cell_name, 'right')
                        case (r, 0) if r == bord_size - 1:
                            graph.add_graph_node_property(id, cell_name, 'bottom')
                            graph.add_graph_node_property(id, cell_name, 'left')
                        case (r, c) if r == bord_size - 1 and c == bord_size - 1:
                            graph.add_graph_node_property(id, cell_name, 'bottom')
                            graph.add_graph_node_property(id, cell_name, 'right')
                        case (0, _):
                            graph.add_graph_node_property(id, cell_name, 'top')
                        case (_, 0):
                            graph.add_graph_node_property(id, cell_name, 'left')
                        case (r, _) if r == bord_size - 1:
                            graph.add_graph_node_property(id, cell_name, 'bottom')
                        case (_, c) if c == bord_size - 1:
                            graph.add_graph_node_property(id, cell_name, 'right')

        # encode:
        graph.encode()



    # create the symbols
    bord_size = len(X_train[0])
    symbols = ['empty', 'red', 'blue']
    for i in range(bord_size):
        symbols.append(f'row_{i}')
        symbols.append(f'col_{i}')
    symbols.extend(['top', 'bottom', 'left', 'right'])
    # symbols.extend(['top_left', 'top_right', 'bottom_left', 'bottom_right'])



    # create the training graphs
    num_training_graphs = len(X_train)
    training_graph = Graphs(
        num_training_graphs,
        symbols=symbols,
        hypervector_bits=args.hypervector_bits,
        hypervector_size=args.hypervector_size,
    )
    configure_graphs(training_graph, X_train)



    # create the test graphs
    num_test_graphs = len(X_test)
    test_graph = Graphs(
        num_test_graphs,
        init_with=training_graph,
    )
    configure_graphs(test_graph, X_test)


    return training_graph, test_graph



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
            f"Time: {epoch_time:.2f}s"
        )



###################################################
################ Print the clauses ################
###################################################
def print_clauses(tm, hypervector_size):
    weights = tm.get_state()[1].reshape(2, -1)
    for i in range(tm.number_of_clauses):
            print("Clause #%d Weights:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
            l = []
            for k in range(hypervector_size * 2):
                if tm.ta_action(0, i, k):
                    if k < hypervector_size:
                        l.append("x%d" % (k))
                    else:
                        l.append("NOT x%d" % (k - hypervector_size))
            print(" AND ".join(l))



###################################################
################ Print the clauses ################
###################################################
# 1. get dataset : get_hex_games
# 2. Create the graphs : create_graphs
# 3. Train the tm : train_tm

args = default_args()

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    max_included_literals=args.max_included_literals,
    number_of_state_bits=args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    double_hashing=args.double_hashing,
    one_hot_encoding=args.one_hot_encoding,
)

print_args(args)
x_train, y_train, x_test, y_test = get_hex_games(0.8, how_many=args.data_size, board_size=args.board_size)
train_graph, test_graph = create_graphs(x_train, x_test, args)
train_tm(tm, train_graph, y_train, test_graph, y_test, args)

print_clauses(tm, args.hypervector_size)
