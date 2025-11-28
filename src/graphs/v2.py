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


    return training_graph, test_graph, symbols



