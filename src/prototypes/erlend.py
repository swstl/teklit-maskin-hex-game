import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--number-of-clauses", default=10000, type=int)
    parser.add_argument("--T", default=15000, type=int)
    parser.add_argument("--s", default=2.5, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

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

def create_hex_graphs_train(boards, labels, num_samples):
    """Convert Hex boards to Graph Tsetlin Machine format - Training"""
    # Create symbols for each position with each player
    symbols = []
    for row in range(7):
        for col in range(7):
            symbols.append(f"P1_{row}_{col}")  # Player 1 at this position
            symbols.append(f"P2_{row}_{col}")  # Player 2 at this position
    
    graphs = Graphs(
        num_samples,
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing,
        one_hot_encoding=args.one_hot_encoding
    )
    
    # Single node per graph (vanilla style)
    for graph_id in range(num_samples):
        graphs.set_number_of_graph_nodes(graph_id, 1)
    
    graphs.prepare_node_configuration()
    
    # Add nodes with NO edges
    for graph_id in range(num_samples):
        graphs.add_graph_node(graph_id, 'Board', 0)
    
    graphs.prepare_edge_configuration()
    
    # Add properties - only for occupied cells!
    Y = np.empty(num_samples, dtype=np.uint32)
    for graph_id in range(num_samples):
        if graph_id % 1000 == 0:
            print(f"{graph_id}/{num_samples}")
            
        board = boards[graph_id]
        for row in range(7):
            for col in range(7):
                cell_value = board[row, col]
                
                # Only add properties for occupied cells
                if cell_value == 1:
                    graphs.add_graph_node_property(graph_id, 'Board', f"P1_{row}_{col}")
                elif cell_value == -1:
                    graphs.add_graph_node_property(graph_id, 'Board', f"P2_{row}_{col}")
        
        # Winner: 1 for player 1, 0 for player 2
        Y[graph_id] = 0 if labels[graph_id] == -1 else 1
    
    graphs.encode()
    
    return graphs, Y

# Load data
print("Loading data...")
boards, labels = load_hex_data('/home/coder/teklit-maskin-hex-game/data/hex_games_1_000_000_size_7.csv')

# Split data  
train_size = 100000  # Use 100k
test_size = 20000     # Use 20k

boards_train = boards[:train_size]
labels_train = labels[:train_size]
boards_test = boards[train_size:train_size+test_size]
labels_test = labels[train_size:train_size+test_size]

# Create training graphs
print("Creating training graphs...")
graphs_train, Y_train = create_hex_graphs_train(boards_train, labels_train, train_size)

print("Training data produced")

# Create test graphs - CRITICAL: init from training graphs!
print("Creating test graphs...")
graphs_test = Graphs(test_size, init_with=graphs_train)

for graph_id in range(test_size):
    graphs_test.set_number_of_graph_nodes(graph_id, 1)

graphs_test.prepare_node_configuration()

for graph_id in range(test_size):
    graphs_test.add_graph_node(graph_id, 'Board', 0)

graphs_test.prepare_edge_configuration()

Y_test = np.empty(test_size, dtype=np.uint32)
for graph_id in range(test_size):
    if graph_id % 1000 == 0:
        print(f"{graph_id}/{test_size}")
        
    board = boards_test[graph_id]
    for row in range(7):
        for col in range(7):
            cell_value = board[row, col]
            
            if cell_value == 1:
                graphs_test.add_graph_node_property(graph_id, 'Board', f"P1_{row}_{col}")
            elif cell_value == -1:
                graphs_test.add_graph_node_property(graph_id, 'Board', f"P2_{row}_{col}")
    
    Y_test[graph_id] = 0 if labels_test[graph_id] == -1 else 1

graphs_test.encode()

print("Testing data produced")

# Initialize Graph Tsetlin Machine
print("Initializing Graph Tsetlin Machine...")
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits=args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    double_hashing=args.double_hashing,
    one_hot_encoding=args.one_hot_encoding,
    grid=(16*13, 1, 1),
    block=(128, 1, 1)
)

# Train and evaluate
print("\nTraining...")

for i in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()
    
    start_testing = time()
    result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()
    
    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()
    
    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

