# engine/bitboard_position.py

import chess
from dataclasses import dataclass


@dataclass
class BitboardPosition:
    """
    Bitboard representation extracted from a python-chess Board.
    Each field is a 64-bit integer (Python int), where bit i corresponds to square i:
    0 = a1, 1 = b1, ..., 63 = h8 (python-chess convention).
    """
    white_pawns: int
    white_knights: int
    white_bishops: int
    white_rooks: int
    white_queens: int
    white_king: int

    black_pawns: int
    black_knights: int
    black_bishops: int
    black_rooks: int
    black_queens: int
    black_king: int

    white_to_move: bool
    castling_rights: int   # python-chess bitmask
    en_passant_square: int  # -1 if none, else 0..63


def from_chess_board(board: chess.Board) -> BitboardPosition:
    """
    Create a BitboardPosition from a python-chess Board.
    """
    return BitboardPosition(
        white_pawns=int(board.pieces(chess.PAWN, chess.WHITE)),
        white_knights=int(board.pieces(chess.KNIGHT, chess.WHITE)),
        white_bishops=int(board.pieces(chess.BISHOP, chess.WHITE)),
        white_rooks=int(board.pieces(chess.ROOK, chess.WHITE)),
        white_queens=int(board.pieces(chess.QUEEN, chess.WHITE)),
        white_king=int(board.pieces(chess.KING, chess.WHITE)),

        black_pawns=int(board.pieces(chess.PAWN, chess.BLACK)),
        black_knights=int(board.pieces(chess.KNIGHT, chess.BLACK)),
        black_bishops=int(board.pieces(chess.BISHOP, chess.BLACK)),
        black_rooks=int(board.pieces(chess.ROOK, chess.BLACK)),
        black_queens=int(board.pieces(chess.QUEEN, chess.BLACK)),
        black_king=int(board.pieces(chess.KING, chess.BLACK)),

        white_to_move=board.turn,
        castling_rights=board.castling_rights,
        en_passant_square=board.ep_square if board.ep_square is not None else -1,
    )
