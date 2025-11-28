from GraphTsetlinMachine.graphs import Graphs

def create_graphs(x_train, x_test, args):
    def create_hex_graphs_train(boards):
        """Convert Hex boards to Graph Tsetlin Machine format - Training"""
        # Create symbols for each position with each player
        symbols = []
        for row in range(7):
            for col in range(7):
                symbols.append(f"P1_{row}_{col}")  # Player 1 at this position
                symbols.append(f"P2_{row}_{col}")  # Player 2 at this position
        
        num_samples = len(boards)
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
            
        
        graphs.encode()
        
        return graphs, symbols


    graphs_train, symbols = create_hex_graphs_train(x_train)

    test_size = len(x_test)
    graphs_test = Graphs(test_size, init_with=graphs_train)

    for graph_id in range(test_size):
        graphs_test.set_number_of_graph_nodes(graph_id, 1)

    graphs_test.prepare_node_configuration()

    for graph_id in range(test_size):
        graphs_test.add_graph_node(graph_id, 'Board', 0)

    graphs_test.prepare_edge_configuration()

    for graph_id in range(test_size):
        if graph_id % 1000 == 0:
            print(f"{graph_id}/{test_size}")
        board = x_test[graph_id]
        for row in range(7):
            for col in range(7):
                cell_value = board[row, col]
                if cell_value == 1:
                    graphs_test.add_graph_node_property(graph_id, 'Board', f"P1_{row}_{col}")
                elif cell_value == -1:
                    graphs_test.add_graph_node_property(graph_id, 'Board', f"P2_{row}_{col}")

    graphs_test.encode()

    return graphs_train, graphs_test, symbols

