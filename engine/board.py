# engine/board.py

import chess
from .bitboard_position import BitboardPosition, from_chess_board


class Board:
    """
    Thin wrapper around python-chess.Board.

    - Used by search and evaluation.
    - Exposes helpful convenience methods.
    - Provides a BitboardPosition view for ML/encoding.
    """

    def __init__(self, fen: str = None):
        if fen is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen)

    def copy(self) -> "Board":
        new_board = Board()
        new_board.board = self.board.copy()
        return new_board

    def generate_legal_moves(self):
        """Return a list of legal moves (python-chess Move objects)."""
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def pop(self) -> chess.Move:
        return self.board.pop()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        """
        "1-0", "0-1", "1/2-1/2" or "*".
        """
        return self.board.result(claim_draw=True)

    def turn(self) -> bool:
        """True if it's White to move, False if Black."""
        return self.board.turn

    def fen(self) -> str:
        return self.board.fen()

    def set_fen(self, fen: str) -> None:
        self.board.set_fen(fen)

    def to_bitboards(self) -> BitboardPosition:
        """Return a BitboardPosition view of the current board."""
        return from_chess_board(self.board)
