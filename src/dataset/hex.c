// Copyright (c) 2024 Ole-Christoffer Granmo
// Modified to accept command line arguments

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <getopt.h>

int BOARD_DIM = 5;
int NUM_GAMES = 1000;

int *neighbors;

struct hex_game {
	int *board;
	int *open_positions;
	int number_of_open_positions;
	int *moves;
	int *connected;
};

void init_neighbors() {
	neighbors = malloc(6 * sizeof(int));
	neighbors[0] = -(BOARD_DIM+2) + 1;
	neighbors[1] = -(BOARD_DIM+2);
	neighbors[2] = -1;
	neighbors[3] = 1;
	neighbors[4] = (BOARD_DIM+2);
	neighbors[5] = (BOARD_DIM+2) - 1;
}

void hg_alloc(struct hex_game *hg)
{
	hg->board = malloc((BOARD_DIM+2)*(BOARD_DIM+2)*2 * sizeof(int));
	hg->open_positions = malloc(BOARD_DIM*BOARD_DIM * sizeof(int));
	hg->moves = malloc(BOARD_DIM*BOARD_DIM * sizeof(int));
	hg->connected = malloc((BOARD_DIM+2)*(BOARD_DIM+2)*2 * sizeof(int));
}

void hg_free(struct hex_game *hg)
{
	free(hg->board);
	free(hg->open_positions);
	free(hg->moves);
	free(hg->connected);
}

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

int hg_get_cell(struct hex_game *hg, int row, int col)
{
	int pos = (row + 1) * (BOARD_DIM + 2) + (col + 1);
	if (hg->board[pos * 2] == 1) {
		return 1;
	} else if (hg->board[pos * 2 + 1] == 1) {
		return -1;
	}
	return 0;
}

void hg_write_csv_row(FILE *f, struct hex_game *hg, int winner)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < BOARD_DIM; ++j) {
			fprintf(f, "%d,", hg_get_cell(hg, i, j));
		}
	}
	fprintf(f, "%d\n", winner == 0 ? 1 : -1);
}

void hg_write_csv_header(FILE *f)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < BOARD_DIM; ++j) {
			fprintf(f, "cell%d_%d,", i, j);
		}
	}
	fprintf(f, "winner\n");
}

void print_usage(char *prog_name) {
	printf("Usage: %s [-g num_games] [-b board_size] [-h]\n", prog_name);
	printf("  -g NUM    Number of games to generate (default: 1000)\n");
	printf("  -b SIZE   Board dimension (default: 5)\n");
	printf("  -h        Show this help message\n");
}

int main(int argc, char *argv[]) {
	int opt;
	
	while ((opt = getopt(argc, argv, "g:b:h")) != -1) {
		switch (opt) {
			case 'g':
				NUM_GAMES = atoi(optarg);
				break;
			case 'b':
				BOARD_DIM = atoi(optarg);
				break;
			case 'h':
				print_usage(argv[0]);
				return 0;
			default:
				print_usage(argv[0]);
				return 1;
		}
	}

	printf("Generating %d games on %dx%d board...\n", NUM_GAMES, BOARD_DIM, BOARD_DIM);

	struct hex_game hg;
	
	srand(time(NULL));
	init_neighbors();
	hg_alloc(&hg);

	char filename[100];
	sprintf(filename, "hex_games_%d_size_%d.csv", NUM_GAMES, BOARD_DIM);

	FILE *f = fopen(filename, "w");
	hg_write_csv_header(f);

	int winner = -1;

	for (int game = 0; game < NUM_GAMES; ++game) {
		hg_init(&hg);

		int player = 0;
		while (!hg_full_board(&hg)) {
			int position = hg_place_piece_randomly(&hg, player);
			
			if (hg_winner(&hg, player, position)) {
				winner = player;
				break;
			}

			player = 1 - player;
		}

		hg_write_csv_row(f, &hg, winner);

		if (game % 100000 == 0 && game != 0) {
			printf("Generated %d games...\n", game);
		}
	}

	fclose(f);
	hg_free(&hg);
	free(neighbors);
	
	printf("Done! Generated %d games. Output: %s\n", NUM_GAMES, filename);
	return 0;
}
