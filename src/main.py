"""
Graph Tsetlin Machine for Hex Game Winner Prediction
FIXED VERSION - handles CUDA memory issues

Board encoding: -1 = Red/Player1, 1 = Blue/Player2, 0 = Empty
"""

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import numpy as np
import pandas as pd
import argparse
from time import time

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--T", default=2000, type=int)
    parser.add_argument("--s", default=5.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=512, type=int)
    parser.add_argument("--hypervector-bits", default=4, type=int)
    parser.add_argument("--message-size", default=512, type=int)
    parser.add_argument("--message-bits", default=4, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


def load_hex_data(csv_path):
    """Load the Hex game CSV data"""
    df = pd.read_csv(csv_path)
    
    # Get board columns
    board_columns = [col for col in df.columns if col.startswith('cell')]
    
    # Determine board size from columns
    max_row = 0
    max_col = 0
    for col in board_columns:
        parts = col.replace('cell', '').split('_')
        r, c = int(parts[0]), int(parts[1])
        max_row = max(max_row, r)
        max_col = max(max_col, c)
    
    board_size = max(max_row, max_col) + 1
    
    # Reshape each row into NxN board
    boards = []
    for _, row in df.iterrows():
        board = np.zeros((board_size, board_size), dtype=int)
        for col in board_columns:
            parts = col.replace('cell', '').split('_')
            r, c = int(parts[0]), int(parts[1])
            board[r, c] = row[col]
        boards.append(board)
    
    labels = df['winner'].values
    return boards, labels, board_size


def get_hex_neighbors(row, col, board_size):
    """Get valid neighboring cells in Hex grid (6 neighbors)."""
    neighbors = []
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            neighbors.append((nr, nc))
    return neighbors


def create_hex_graphs(boards, board_size, args, init_graphs=None):
    """
    Create Graph TM compatible graphs from Hex board states.
    SIMPLIFIED version with fewer symbols to avoid CUDA issues.
    """
    num_graphs = len(boards)
    
    # Keep symbols minimal to avoid memory issues
    symbols = ['Red', 'Blue', 'Empty']
    
    # Row and column markers
    for i in range(board_size):
        symbols.append(f'Row{i}')
        symbols.append(f'Col{i}')
    
    # Edge markers (CRITICAL for Hex)
    symbols.extend(['TopEdge', 'BottomEdge', 'LeftEdge', 'RightEdge'])
    
    print(f"Total symbols: {len(symbols)}")
    
    if init_graphs is None:
        graphs = Graphs(
            num_graphs,
            symbols=symbols,
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits
        )
    else:
        graphs = Graphs(num_graphs, init_with=init_graphs)
    
    num_cells = board_size * board_size
    
    # Set number of nodes for each graph
    for graph_id in range(num_graphs):
        graphs.set_number_of_graph_nodes(graph_id, num_cells)
    
    graphs.prepare_node_configuration()
    
    # Add nodes to each graph
    for graph_id in range(num_graphs):
        for r in range(board_size):
            for c in range(board_size):
                cell_name = f'Cell_{r}_{c}'
                num_edges = len(get_hex_neighbors(r, c, board_size))
                graphs.add_graph_node(graph_id, cell_name, num_edges)
    
    graphs.prepare_edge_configuration()
    
    # Add edges between adjacent cells
    for graph_id in range(num_graphs):
        for r in range(board_size):
            for c in range(board_size):
                cell_name = f'Cell_{r}_{c}'
                neighbors = get_hex_neighbors(r, c, board_size)
                
                for nr, nc in neighbors:
                    neighbor_name = f'Cell_{nr}_{nc}'
                    
                    # Use simple edge type to reduce complexity
                    edge_type = "Adjacent"
                    
                    graphs.add_graph_node_edge(graph_id, cell_name, neighbor_name, edge_type)
    
    # Add properties to nodes
    for graph_id in range(num_graphs):
        board = boards[graph_id]
        for r in range(board_size):
            for c in range(board_size):
                cell_name = f'Cell_{r}_{c}'
                
                # Piece state
                if board[r, c] == -1:
                    graphs.add_graph_node_property(graph_id, cell_name, 'Red')
                elif board[r, c] == 1:
                    graphs.add_graph_node_property(graph_id, cell_name, 'Blue')
                else:
                    graphs.add_graph_node_property(graph_id, cell_name, 'Empty')
                
                # Position
                graphs.add_graph_node_property(graph_id, cell_name, f'Row{r}')
                graphs.add_graph_node_property(graph_id, cell_name, f'Col{c}')
                
                # Edge markers
                if r == 0:
                    graphs.add_graph_node_property(graph_id, cell_name, 'TopEdge')
                if r == board_size - 1:
                    graphs.add_graph_node_property(graph_id, cell_name, 'BottomEdge')
                if c == 0:
                    graphs.add_graph_node_property(graph_id, cell_name, 'LeftEdge')
                if c == board_size - 1:
                    graphs.add_graph_node_property(graph_id, cell_name, 'RightEdge')
    
    graphs.encode()
    return graphs


def convert_labels(labels):
    """Convert labels to 0/1 format for TM."""
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    print(f"Original label values: {unique_labels}")
    
    if -1 in unique_labels:
        converted = ((labels + 1) // 2).astype(np.uint32)
    else:
        converted = labels.astype(np.uint32)
    
    print(f"Converted: Class 0 = {np.sum(converted==0)}, Class 1 = {np.sum(converted==1)}")
    
    return converted


def get_recommended_params(board_size):
    """Get recommended hyperparameters based on board size."""
    if board_size <= 3:
        return {
            'epochs': 100,
            'number_of_clauses': 1000,
            'T': 2000,
            's': 5.0,
            'depth': 2,
            'hypervector_size': 512,
            'hypervector_bits': 4,
            'message_size': 512,
            'message_bits': 4,
            'max_included_literals': 32
        }
    elif board_size <= 5:
        return {
            'epochs': 150,
            'number_of_clauses': 2000,
            'T': 4000,
            's': 8.0,
            'depth': 3,
            'hypervector_size': 512,
            'hypervector_bits': 4,
            'message_size': 512,
            'message_bits': 4,
            'max_included_literals': 64
        }
    else:  # 7x7 and larger
        return {
            'epochs': 250,
            'number_of_clauses': 4000,
            'T': 8000,
            's': 10.0,
            'depth': 4,
            'hypervector_size': 1024,
            'hypervector_bits': 8,
            'message_size': 1024,
            'message_bits': 8,
            'max_included_literals': 128
        }


def train_hex_predictor(train_csv, test_csv=None, args=None, auto_params=True):
    """Train Graph TM to predict Hex game winners."""
    
    # Load training data first to get board size
    print("Loading training data...")
    boards, labels, board_size = load_hex_data(train_csv)
    
    print(f"Board size: {board_size}x{board_size}")
    print(f"Number of samples: {len(boards)}")
    
    # Auto-select parameters if requested
    if auto_params or args is None:
        params = get_recommended_params(board_size)
        args = default_args(**params)
        print(f"\nUsing auto-selected parameters for {board_size}x{board_size} board")
    
    # Convert labels
    labels = convert_labels(labels)
    
    # Split into train/test
    if test_csv is None:
        split_idx = int(len(boards) * 0.8)
        indices = np.random.permutation(len(boards))
        
        train_boards = [boards[i] for i in indices[:split_idx]]
        test_boards = [boards[i] for i in indices[split_idx:]]
        Y_train = labels[indices[:split_idx]]
        Y_test = labels[indices[split_idx:]]
    else:
        train_boards = boards
        Y_train = labels
        
        test_boards, test_labels, _ = load_hex_data(test_csv)
        Y_test = convert_labels(test_labels)
    
    print(f"Training samples: {len(train_boards)}")
    print(f"Test samples: {len(test_boards)}")
    
    # Create graphs
    print("\nCreating training graphs...")
    graphs_train = create_hex_graphs(train_boards, board_size, args)
    
    print("Creating test graphs...")
    graphs_test = create_hex_graphs(test_boards, board_size, args, graphs_train)
    
    # Initialize Graph TM
    print("\nInitializing Graph Tsetlin Machine...")
    print(f"  Clauses: {args.number_of_clauses}")
    print(f"  T: {args.T}")
    print(f"  s: {args.s}")
    print(f"  Depth: {args.depth}")
    print(f"  Hypervector size: {args.hypervector_size}")
    print(f"  Message size: {args.message_size}")
    
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing=args.double_hashing
    )
    
    # Training loop
    print("\nTraining...")
    print(f"{'Epoch':>5} {'Train%':>8} {'Test%':>8} {'Time(s)':>10}")
    print("-" * 35)
    
    best_test_acc = 0.0
    
    for epoch in range(args.epochs):
        start = time()
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
        train_time = time() - start
        
        # Evaluate
        test_pred = tm.predict(graphs_test)
        test_acc = 100 * (test_pred == Y_test).mean()
        
        train_pred = tm.predict(graphs_train)
        train_acc = 100 * (train_pred == Y_train).mean()
        
        print(f"{epoch:>5} {train_acc:>8.2f} {test_acc:>8.2f} {train_time:>10.2f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Early stopping at 100%
        if test_acc == 100.0:
            print(f"\n*** 100% accuracy reached at epoch {epoch}! ***")
            break
        
        # Warn if stuck
        if epoch == 30 and best_test_acc < 55:
            print("\n*** Warning: Model may be stuck. Try increasing clauses or s. ***\n")
    
    print(f"\nBest test accuracy: {best_test_acc:.2f}%")
    print(f"Number of clauses: {args.number_of_clauses}")
    
    return tm, best_test_acc, graphs_train


# Example usage
if __name__ == "__main__":
    tm, accuracy, graphs_train = train_hex_predictor(
        'data/hex_games_1000000_size_5.csv',
        auto_params=True 
    )
