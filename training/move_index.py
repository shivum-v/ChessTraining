import chess

uci_to_idx = {}
idx_to_uci = []

def build_move_index():
    """Builds the same move index you used in the notebook."""
    global uci_to_idx, idx_to_uci
    all_moves = []
    for a in range(64):
        for b in range(64):
            if a != b:
                all_moves.append(chess.Move(a, b))
    promos = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for a in range(64):
        rank = a // 8
        if rank == 6:
            for to_sq in range(56, 64):
                for p in promos:
                    all_moves.append(chess.Move(a, to_sq, p))

    all_moves = sorted(all_moves, key=lambda m: m.uci())
    idx_to_uci = [m.uci() for m in all_moves]
    uci_to_idx = {m.uci(): i for i, m in enumerate(all_moves)}
    print(f"[INFO] Move index built with {len(idx_to_uci)} actions.")

# build on import for convenience (same as notebook)
build_move_index()

def move_to_index(move):
    return uci_to_idx.get(move.uci(), 0)

def index_to_uci(i):
    return idx_to_uci[i]

def num_actions():
    return len(idx_to_uci)