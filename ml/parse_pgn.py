# ml/parse_pgn.py

from pathlib import Path
from typing import Iterator, Tuple

import chess
import chess.pgn
import numpy as np

from .encoding import encode_position


def iterate_games(pgn_path: str | Path) -> Iterator[chess.pgn.Game]:
    pgn_path = Path(pgn_path)
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def game_result_value(game: chess.pgn.Game) -> float:
    """
    Returns result from White's perspective:
      1.0  -> White wins
      0.0  -> draw
      -1.0 -> Black wins
    """
    res = game.headers.get("Result", "*")
    if res == "1-0":
        return 1.0
    elif res == "0-1":
        return -1.0
    elif res == "1/2-1/2":
        return 0.0
    else:
        # unfinished or weird; skip
        return None


def generate_value_samples_from_pgn(
    pgn_path: str | Path,
    max_positions: int | None = None
) -> Iterator[Tuple[np.ndarray, float]]:
    """
    Yield (position_tensor, value_label) pairs.
    value_label is from side-to-move's perspective in that position, in [-1, 1].
    """
    count = 0
    for game in iterate_games(pgn_path):
        game_value_white = game_result_value(game)
        if game_value_white is None:
            continue  # skip unfinished games

        board = game.board()

        # For each ply in the game, we take the position *before* the move
        # and label it based on side to move.
        node = game
        while not node.is_end():
            # Position before making the next move
            pos_value = game_value_white
            # If it's Black to move, flip sign (result from that side's perspective)
            if board.turn == chess.BLACK:
                pos_value = -pos_value

            pos_tensor = encode_position(board)  # [C,8,8]

            yield pos_tensor, pos_value
            count += 1
            if max_positions is not None and count >= max_positions:
                return

            # Advance one ply
            next_node = node.variation(0)
            move = next_node.move
            board.push(move)
            node = next_node
