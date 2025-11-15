# training/encoder.py
import numpy as np
import torch
import chess

def encode_board(board: chess.Board, device=None):
    """
    Convert board to tensor (12 x 8 x 8) as in your notebook.
    Returns a torch.FloatTensor on the CPU unless device specified.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        channel_offset = 6 if piece.color == chess.WHITE else 0
        planes[channel_offset + (piece.piece_type - 1), row, col] = 1.0
    t = torch.tensor(planes, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
