# scripts/trim_pgn.py

import argparse
from pathlib import Path
import chess.pgn


def trim_pgn(input_path: str, output_path: str, max_games: int):
    in_path = Path(input_path)
    out_path = Path(output_path)

    with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        count = 0
        while count < max_games:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            fout.write(str(game))
            fout.write("\n\n")
            count += 1

    print(f"Wrote {count} games to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_games", type=int, default=5000)
    args = parser.parse_args()

    trim_pgn(args.input, args.output, args.max_games)


if __name__ == "__main__":
    main()
