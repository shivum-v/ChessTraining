# training/self_play.py
import chess
import numpy as np
from .mcts import mcts_search

def self_play_episode(model, device, simulations=25):
    """
    Plays a single self-play game using model+MCTS.
    Returns arrays: states, pis, rewards matching your old format.
    """
    board = chess.Board()
    states, pis = [], []

    while not board.is_game_over():
        move, root = mcts_search(board, model=model, device=device, simulations=simulations)
        # collect state
        # encode on demand in train loop (we'll return raw boards for flexibility)
        # but to match notebook, we return tensors/arrays -> we'll encode states here
        # For simplicity we encode states as numpy arrays (12x8x8)
        from .encoder import encode_board
        state_tensor = encode_board(board, device=None).cpu().numpy()
        states.append(state_tensor)

        # build pi: one-hot on chosen move (as in notebook)
        from .move_index import move_to_index, num_actions
        pi = np.zeros(num_actions(), dtype=np.float32)
        pi[move_to_index(move)] = 1.0
        pis.append(pi)

        board.push(move)

    # final reward: +1, -1, 0 (from White perspective)
    result = board.result()
    if result == "1-0":
        r = 1
    elif result == "0-1":
        r = -1
    else:
        r = 0
    rewards = [r] * len(states)
    return np.array(states), np.array(pis), np.array(rewards)
