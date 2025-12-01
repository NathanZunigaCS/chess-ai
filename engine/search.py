# engine/search.py

import math
import random
import time
from enum import Enum

import chess

from .board import Board
from .eval import evaluate, MATE_SCORE


class NodeType(Enum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


class TTEntry:
    __slots__ = ("depth", "score", "node_type", "best_move")

    def __init__(self, depth: int, score: float, node_type: NodeType, best_move: chess.Move):
        self.depth = depth
        self.score = score
        self.node_type = node_type
        self.best_move = best_move


# Very simple transposition table; keyed by FEN. Good enough for base engine.
TRANSPOSITION_TABLE: dict[str, TTEntry] = {}


def clear_tt():
    TRANSPOSITION_TABLE.clear()


def search_best_move(board: Board, depth: int = 3, max_time: float = None) -> chess.Move | None:
    """
    Entry point: iterative deepening alpha-beta search.
    - depth: maximum depth (in plies).
    - max_time: optional time limit in seconds for the whole search.
    """

    start_time = time.time()
    best_move = None
    best_score = -math.inf if board.turn() else math.inf

    # Iterative deepening from 1..depth
    for current_depth in range(1, depth + 1):
        # Check time
        if max_time is not None and time.time() - start_time > max_time:
            break

        # Negamax search: score is always from side-to-move's perspective
        score, move = _negamax_root(board, current_depth, start_time, max_time)

        if move is not None:
            best_move = move
            best_score = score

    # Fallback: random legal move if something went wrong
    if best_move is None:
        legal_moves = board.generate_legal_moves()
        if not legal_moves:
            return None
        best_move = random.choice(legal_moves)

    return best_move


def _negamax_root(board: Board, depth: int, start_time: float, max_time: float | None):
    alpha = -math.inf
    beta = math.inf
    best_move = None
    best_score = -math.inf

    legal_moves = board.generate_legal_moves()
    if not legal_moves:
        return evaluate(board), None

    # Simple move ordering: shuffle for now; we'll improve with TT + captures first
    ordered_moves = _order_moves(board, legal_moves)

    color = 1 if board.turn() else -1  # +1 for White, -1 for Black

    for move in ordered_moves:
        if max_time is not None and time.time() - start_time > max_time:
            break

        child = board.copy()
        child.push(move)

        score = -_negamax(child, depth - 1, -beta, -alpha, -color, start_time, max_time)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score, best_move


def _negamax(board: Board, depth: int, alpha: float, beta: float, color: int,
             start_time: float, max_time: float | None) -> float:
    """
    Negamax with alpha-beta pruning and a tiny quiescence search.
    color = +1 if this node is from White's perspective, -1 for Black.
    """

    # Time check
    if max_time is not None and time.time() - start_time > max_time:
        # Emergency cut: return static eval; not great but prevents flagging.
        return color * evaluate(board)

    b = board.board
    fen_key = b.fen()

    # Transposition table lookup
    entry = TRANSPOSITION_TABLE.get(fen_key)
    if entry is not None and entry.depth >= depth:
        if entry.node_type == NodeType.EXACT:
            return entry.score
        elif entry.node_type == NodeType.LOWERBOUND:
            alpha = max(alpha, entry.score)
        elif entry.node_type == NodeType.UPPERBOUND:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    # Leaf node / quiescence
    if depth == 0:
        return _quiescence(board, alpha, beta, color, start_time, max_time)

    legal_moves = board.generate_legal_moves()
    if not legal_moves:
        # Checkmate or stalemate
        if b.is_checkmate():
            # From perspective of side to move: mate is bad
            return -color * MATE_SCORE
        else:
            # Stalemate
            return 0.0

    value = -math.inf
    best_move = None

    ordered_moves = _order_moves(board, legal_moves, tt_move=entry.best_move if entry else None)

    for move in ordered_moves:
        child = board.copy()
        child.push(move)

        score = -_negamax(child, depth - 1, -beta, -alpha, -color, start_time, max_time)

        if score > value:
            value = score
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # Store in TT
    node_type = NodeType.EXACT
    if value <= alpha:
        node_type = NodeType.UPPERBOUND
    elif value >= beta:
        node_type = NodeType.LOWERBOUND

    TRANSPOSITION_TABLE[fen_key] = TTEntry(depth, value, node_type, best_move)

    return value


def _quiescence(board: Board, alpha: float, beta: float, color: int,
                start_time: float, max_time: float | None) -> float:
    """
    Very small quiescence search: only consider capture moves.
    Helps reduce horizon effects on obvious captures.
    """
    # Time check
    if max_time is not None and time.time() - start_time > max_time:
        return color * evaluate(board)

    stand_pat = color * evaluate(board)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    capture_moves = [m for m in board.generate_legal_moves() if board.board.is_capture(m)]
    if not capture_moves:
        return stand_pat

    for move in capture_moves:
        child = board.copy()
        child.push(move)
        score = -_quiescence(child, -beta, -alpha, -color, start_time, max_time)

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha


def _order_moves(board: Board, moves, tt_move: chess.Move | None = None):
    """
    Basic move ordering:
    - TT move first (if any).
    - Then captures.
    - Then the rest.
    """

    captures = []
    quiets = []

    for m in moves:
        if tt_move is not None and m == tt_move:
            # Put TT move at front later
            continue
        if board.board.is_capture(m):
            captures.append(m)
        else:
            quiets.append(m)

    random.shuffle(captures)
    random.shuffle(quiets)

    ordered = []
    if tt_move is not None:
        ordered.append(tt_move)
    ordered.extend(captures)
    ordered.extend(quiets)
    return ordered
