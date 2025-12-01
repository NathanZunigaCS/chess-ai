# ml/encoding.py

import numpy as np
import chess

from engine.bitboard_position import from_chess_board

BOARD_SIZE = 8


def bitboard_to_plane(bb: int) -> np.ndarray:
    """
    Convert a 64-bit bitboard into an 8x8 numpy array of 0/1.
    Square mapping follows python-chess: 0=a1, ..., 63=h8.
    We'll use [rank, file] with rank 0 = first rank (a1..h1).
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square in range(64):
        if (bb >> square) & 1:
            rank = square // 8  # 0..7
            file = square % 8   # 0..7
            plane[rank, file] = 1.0
    return plane


def encode_position(board: chess.Board) -> np.ndarray:
    """
    Encode a python-chess Board into a [C, 8, 8] tensor.
    Channels:
      0-5:  white P,N,B,R,Q,K
      6-11: black P,N,B,R,Q,K
      12:   side to move
      13-16: castling rights (WK, WQ, BK, BQ)
      17:   en passant file
    """
    bb = from_chess_board(board)

    planes = []

    # 0-5: white pieces
    planes.append(bitboard_to_plane(bb.white_pawns))
    planes.append(bitboard_to_plane(bb.white_knights))
    planes.append(bitboard_to_plane(bb.white_bishops))
    planes.append(bitboard_to_plane(bb.white_rooks))
    planes.append(bitboard_to_plane(bb.white_queens))
    planes.append(bitboard_to_plane(bb.white_king))

    # 6-11: black pieces
    planes.append(bitboard_to_plane(bb.black_pawns))
    planes.append(bitboard_to_plane(bb.black_knights))
    planes.append(bitboard_to_plane(bb.black_bishops))
    planes.append(bitboard_to_plane(bb.black_rooks))
    planes.append(bitboard_to_plane(bb.black_queens))
    planes.append(bitboard_to_plane(bb.black_king))

    # 12: side to move
    stm_plane = np.full((BOARD_SIZE, BOARD_SIZE),
                        1.0 if bb.white_to_move else 0.0,
                        dtype=np.float32)
    planes.append(stm_plane)

    # 13-16: castling rights
    wk = np.full((BOARD_SIZE, BOARD_SIZE),
                 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
                 dtype=np.float32)
    wq = np.full((BOARD_SIZE, BOARD_SIZE),
                 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
                 dtype=np.float32)
    bk = np.full((BOARD_SIZE, BOARD_SIZE),
                 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
                 dtype=np.float32)
    bq = np.full((BOARD_SIZE, BOARD_SIZE),
                 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
                 dtype=np.float32)
    planes.extend([wk, wq, bk, bq])

    # 17: en passant file
    ep_plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bb.en_passant_square != -1:
        ep_file = bb.en_passant_square % 8
        ep_plane[:, ep_file] = 1.0
    planes.append(ep_plane)

    return np.stack(planes, axis=0)  # [18, 8, 8]
