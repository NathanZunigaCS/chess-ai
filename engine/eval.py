# engine/eval.py

import chess
from .board import Board

# Basic material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # don't score king directly
}

MATE_SCORE = 100000  # large value for checkmates


# Very simple piece-square tables (from White's perspective).
# These are NOT tuned; they're just a decent starting baseline.
# Index: 0 = a1, ..., 63 = h8 (python-chess). We'll mirror for Black.
PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
      0,   0,   5,  10,  10,   5,   0,   0,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
     10,  10,  10,  10,  10,  10,  10,  10,
      0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_PST_MID = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

KING_PST_END = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]


def _is_endgame(b: chess.Board) -> bool:
    """Very crude endgame detection: assume endgame when total queens+rooks is low."""
    queens = len(b.pieces(chess.QUEEN, chess.WHITE)) + len(b.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(b.pieces(chess.ROOK, chess.WHITE)) + len(b.pieces(chess.ROOK, chess.BLACK))
    return queens == 0 or (queens == 1 and rooks <= 1)


def _pst_value(piece_type: chess.PieceType, square: chess.Square, color: bool, endgame: bool) -> int:
    """
    Get PST value from the appropriate table.
    color = chess.WHITE or chess.BLACK.
    square index is in 0..63.
    """
    if piece_type == chess.PAWN:
        table = PAWN_PST
    elif piece_type == chess.KNIGHT:
        table = KNIGHT_PST
    elif piece_type == chess.BISHOP:
        table = BISHOP_PST
    elif piece_type == chess.ROOK:
        table = ROOK_PST
    elif piece_type == chess.QUEEN:
        table = QUEEN_PST
    elif piece_type == chess.KING:
        table = KING_PST_END if endgame else KING_PST_MID
    else:
        return 0

    # For black, mirror the board vertically
    idx = square if color == chess.WHITE else chess.square_mirror(square)
    return table[idx]


def evaluate(board: Board) -> float:
    """
    Static evaluation function in centipawns.
    Positive = better for White, negative = better for Black.

    This is where you'll later plug in a neural network:
    - You could combine NN output with this score.
    - Or replace this function entirely with NN eval.
    """
    b = board.board

    # Checkmate / stalemate handling:
    if b.is_game_over():
        result = b.result(claim_draw=True)
        if result == "1-0":
            return MATE_SCORE
        elif result == "0-1":
            return -MATE_SCORE
        else:
            return 0.0

    endgame = _is_endgame(b)

    score = 0

    # Material + PST
    for piece_type, value in PIECE_VALUES.items():
        for square in b.pieces(piece_type, chess.WHITE):
            score += value
            score += _pst_value(piece_type, square, chess.WHITE, endgame)
        for square in b.pieces(piece_type, chess.BLACK):
            score -= value
            score -= _pst_value(piece_type, square, chess.BLACK, endgame)

    # Slight bonus for side to move
    if b.turn == chess.WHITE:
        score += 10
    else:
        score -= 10

    return float(score)
