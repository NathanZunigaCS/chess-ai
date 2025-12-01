import os
import sys
import json
import threading
import requests
import chess

from engine.board import Board
from engine.search import search_best_move

LICHESS_API_URL = "https://lichess.org"


def auth_headers():
    token = os.getenv("LICHESS_BOT_TOKEN")
    if not token:
        print("ERROR: LICHESS_BOT_TOKEN environment variable not set.")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/x-ndjson",
    }


def main():
    print("Starting Lichess bot event stream...")
    session = requests.Session()
    response = session.get(
        f"{LICHESS_API_URL}/api/stream/event",
        headers=auth_headers(),
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        try:
            event = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        if event.get("type") == "challenge":
            handle_challenge(event, session)
        elif event.get("type") == "gameStart":
            game_id = event["game"]["gameId"]
            threading.Thread(
                target=handle_game, args=(game_id, session), daemon=True
            ).start()


def handle_challenge(event, session):
    challenge_id = event["challenge"]["id"]
    print(f"Accepting challenge {challenge_id}")
    r = session.post(
        f"{LICHESS_API_URL}/api/challenge/{challenge_id}/accept",
        headers=auth_headers(),
    )
    print("Accept status:", r.status_code)


def handle_game(game_id: str, session: requests.Session):
    print(f"Streaming game {game_id}")
    r = session.get(
        f"{LICHESS_API_URL}/api/bot/game/stream/{game_id}",
        headers=auth_headers(),
        stream=True,
    )
    r.raise_for_status()

    board = chess.Board()
    my_color = None

    for line in r.iter_lines():
        if not line:
            continue

        try:
            event = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        t = event.get("type")

        if t == "gameFull":
            white_id = event["white"]["id"]
            black_id = event["black"]["id"]
            my_id = os.getenv("LICHESS_BOT_ID")

            my_color = "white" if white_id == my_id else "black"

            moves_str = event["state"].get("moves", "")
            apply_moves(board, moves_str)

        elif t == "gameState":
            moves_str = event.get("moves", "")
            apply_moves(board, moves_str)

            if my_color is None:
                continue

            is_my_turn = (
                board.turn == chess.WHITE and my_color == "white"
            ) or (
                board.turn == chess.BLACK and my_color == "black"
            )

            if is_my_turn and not board.is_game_over():
                move = pick_move(board)
                if move:
                    send_move(game_id, move, session)

        elif t == "chatLine":
            pass

    print(f"Game {game_id} ended.")


def apply_moves(board: chess.Board, moves_str: str):
    board.reset()
    if not moves_str:
        return
    for uci in moves_str.split():
        board.push(chess.Move.from_uci(uci))


def pick_move(board: chess.Board):
    engine_board = Board(board.fen())
    return search_best_move(engine_board, depth=3)


def send_move(game_id: str, move: chess.Move, session):
    uci = move.uci()
    print(f"Playing {uci} in game {game_id}")

    r = session.post(
        f"{LICHESS_API_URL}/api/bot/game/{game_id}/move/{uci}",
        headers=auth_headers(),
    )

    if r.status_code != 200:
        print("Move rejected:", r.status_code, r.text)


if __name__ == "__main__":
    main()
