from GraphTsetlinMachine.graphs import Graphs

def create_graphs(X_train, X_test, args):
    """
    V5: Back to basics with COMPLETE information
    
    Key insight: EVERY cell must have occupancy info (red/blue/empty)
    - Empty cells are crucial for path detection
    - TM needs to reason about "no red blocking here" = potential blue path
    """
    print(f"Creating v5 graphs - training: {len(X_train)}, test: {len(X_test)}...")
    
    def get_neighbours(row, col, board_size):
        """Get all 6 hex neighbors"""
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbours = []
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                neighbours.append((nr, nc))
        return neighbours

    def configure_graphs(graph, X):
        board_size = len(X[0])
        num_graphs = len(X)
        cells = board_size * board_size
        
        # Set number of nodes
        for id in range(num_graphs):
            graph.set_number_of_graph_nodes(id, cells)

        graph.prepare_node_configuration()
        for id in range(num_graphs):
            for row in range(board_size):
                for col in range(board_size):
                    cell_name = f'cell_{row}_{col}'
                    num_edges = len(get_neighbours(row, col, board_size))
                    graph.add_graph_node(id, cell_name, num_edges)

        # Add edges
        graph.prepare_edge_configuration()
        for id in range(num_graphs):
            board = X[id]
            for row in range(board_size):
                for col in range(board_size):
                    cell_name = f'cell_{row}_{col}'

                    # Add neighbor edges
                    for nr, nc in get_neighbours(row, col, board_size):
                        n_name = f'cell_{nr}_{nc}'
                        graph.add_graph_node_edge(id, cell_name, n_name, 'neighbour')

                    # CRITICAL: ALWAYS add occupancy - every cell gets ONE state
                    cell_value = board[row, col]
                    if cell_value == -1:
                        graph.add_graph_node_property(id, cell_name, 'red')
                    elif cell_value == 1:
                        graph.add_graph_node_property(id, cell_name, 'blue')
                    else:  # cell_value == 0
                        graph.add_graph_node_property(id, cell_name, 'empty')

                    # Position encoding
                    graph.add_graph_node_property(id, cell_name, f'row_{row}')
                    graph.add_graph_node_property(id, cell_name, f'col_{col}')

                    # Border markers - critical for winning condition
                    if row == 0:
                        graph.add_graph_node_property(id, cell_name, 'top')
                    if row == board_size - 1:
                        graph.add_graph_node_property(id, cell_name, 'bottom')
                    if col == 0:
                        graph.add_graph_node_property(id, cell_name, 'left')
                    if col == board_size - 1:
                        graph.add_graph_node_property(id, cell_name, 'right')

        graph.encode()

    # Create symbols
    board_size = len(X_train[0])
    symbols = ['empty', 'red', 'blue', 'top', 'bottom', 'left', 'right']
    
    # Add row and column symbols
    for i in range(board_size):
        symbols.append(f'row_{i}')
        symbols.append(f'col_{i}')

    # Create graphs
    training_graph = Graphs(
        len(X_train),
        symbols=symbols,
        hypervector_bits=args.hypervector_bits,
        hypervector_size=args.hypervector_size,
    )
    configure_graphs(training_graph, X_train)

    test_graph = Graphs(
        len(X_test),
        init_with=training_graph,
    )
    configure_graphs(test_graph, X_test)

    return training_graph, test_graph, symbols