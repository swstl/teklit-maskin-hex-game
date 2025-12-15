from GraphTsetlinMachine.graphs import Graphs

def create_graphs(X_train, X_test, args):
    print(f"Creating graphs on dataset of length training: {len(X_train)}, and test: {len(X_test)}...")
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
        cells = bord_size * bord_size

        for id in range(num_graphs):
            graph.set_number_of_graph_nodes(id, cells)

        graph.prepare_node_configuration()
        for id in range(num_graphs):
            for row in range(bord_size):
                for col in range(bord_size):
                    cell_name = f'cell_{row}_{col}'
                    num_edges = len(get_neighbours(row, col, bord_size))
                    graph.add_graph_node(id, cell_name, num_edges)

        graph.prepare_edge_configuration()
        for id in range(num_graphs):
            board = X[id]
            for row in range(bord_size):
                for col in range(bord_size):
                    cell_name = f'cell_{row}_{col}'

                    # Add neighbors (for message passing)
                    neighbours = get_neighbours(row, col, bord_size)
                    for nr, nc in neighbours:
                        n_name = f'cell_{nr}_{nc}'
                        graph.add_graph_node_edge(id, cell_name, n_name, 'neighbour')

                    # Node color
                    cell_value = board[row, col]
                    if cell_value == -1:
                        graph.add_graph_node_property(id, cell_name, 'red')
                    elif cell_value == 0:
                        graph.add_graph_node_property(id, cell_name, 'empty')
                    else:
                        graph.add_graph_node_property(id, cell_name, 'blue')

                    # ONLY mark borders that matter for winning
                    # Red connects left-right, Blue connects top-bottom
                    if col == 0 or col == bord_size - 1:
                        graph.add_graph_node_property(id, cell_name, 'border_red')
                    if row == 0 or row == bord_size - 1:
                        graph.add_graph_node_property(id, cell_name, 'border_blue')

        graph.encode()



    # create the symbols
    symbols = [
        'empty', 'red', 'blue',
        'border_red', 'border_blue',
    ]

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


    return training_graph, test_graph, symbols



