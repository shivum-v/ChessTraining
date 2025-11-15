# training/mcts.py
import numpy as np
from .encoder import encode_board
from .move_index import move_to_index
import torch

class Node:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

def mcts_search(board, model, device, simulations=25, c_puct=1.0):
    """
    Runs MCTS starting from `board`. Uses `model` to evaluate leaves.
    Returns the selected move (argmax visits) and the root node.
    """
    root = Node()

    for _ in range(simulations):
        node = root
        sim_board = board.copy()

        # Selection
        while node.children:
            move, node = max(
                node.children.items(),
                key=lambda kv: kv[1].Q + c_puct * kv[1].P * (np.sqrt(node.N + 1e-8) / (1 + kv[1].N))
            )
            sim_board.push(move)

        # Evaluate leaf with NN
        state = encode_board(sim_board, device=device).unsqueeze(0)  # shape (1,C,8,8)
        with torch.no_grad():
            log_probs, value = model(state)
        probs = log_probs.exp().cpu().numpy()[0]

        # Mask legal moves and normalize priors
        legal_priors = {}
        s = 0.0
        for mv in sim_board.legal_moves:
            p = float(probs[move_to_index(mv)])
            legal_priors[mv] = p
            s += p
        if s <= 0:
            # fallback uniform
            k = len(legal_priors)
            for mv in legal_priors:
                legal_priors[mv] = 1.0 / (k + 1e-9)
        else:
            for mv in legal_priors:
                legal_priors[mv] /= (s + 1e-9)

        # Expansion
        for mv, prior in legal_priors.items():
            node.children[mv] = Node(parent=node, prior=prior)

        # Backprop
        v = float(value.item())
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q = cur.W / cur.N
            v = -v
            cur = cur.parent

    # choose action with highest visits
    if not root.children:
        raise RuntimeError("MCTS root had no children (no legal moves?)")
    moves, nodes = zip(*root.children.items())
    visits = [n.N for n in nodes]
    best_move = moves[int(np.argmax(visits))]
    return best_move, root
