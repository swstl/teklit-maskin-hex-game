from GraphTsetlinMachine.graphs import Graphs

def create_graphs(X_train, X_test, args):
    print(f"Creating graphs on dataset of length training: {len(X_train)}, and test: {len(X_test)}...")
    board_size = len(X_train[0])

    # Create symbols for each position and color
    symbols = []
    for row in range(board_size):
        for col in range(board_size):
            symbols.append(f"red_{row}_{col}")
            symbols.append(f"blue_{row}_{col}")

    # Total symbols: board_size² × 2 colors
    # For 5×5: 50 symbols
    # For 16×16: 512 symbols
 
    num_training_graphs = len(X_train)
    training_graph = Graphs(
        num_training_graphs,
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
    )

    # Each graph has just ONE node
    for id in range(num_training_graphs):
        training_graph.set_number_of_graph_nodes(id, 1)

    training_graph.prepare_node_configuration()

    for id in range(num_training_graphs):
        training_graph.add_graph_node(id, 'board', 0)  # 0 edges

    training_graph.prepare_edge_configuration()

    # Add properties for occupied positions
    for id in range(num_training_graphs):
        board = X_train[id]
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == -1:  # Red
                    training_graph.add_graph_node_property(id, 'board', f"red_{row}_{col}")
                elif board[row, col] == 1:  # Blue
                    training_graph.add_graph_node_property(id, 'board', f"blue_{row}_{col}")

    training_graph.encode()

    # Create test graph
    num_test_graphs = len(X_test)
    test_graph = Graphs(num_test_graphs, init_with=training_graph)

    # Each test graph has just ONE node
    for id in range(num_test_graphs):
        test_graph.set_number_of_graph_nodes(id, 1)

    test_graph.prepare_node_configuration()

    for id in range(num_test_graphs):
        test_graph.add_graph_node(id, 'board', 0)  # 0 edges

    test_graph.prepare_edge_configuration()

    # Add properties for occupied positions in test set
    for id in range(num_test_graphs):
        board = X_test[id]
        for row in range(board_size):
            for col in range(board_size):
                if board[row, col] == -1:  # Red
                    test_graph.add_graph_node_property(id, 'board', f"red_{row}_{col}")
                elif board[row, col] == 1:  # Blue
                    test_graph.add_graph_node_property(id, 'board', f"blue_{row}_{col}")

    test_graph.encode()

    return training_graph, test_graph, symbols
