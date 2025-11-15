# training/train.py
import torch
import torch.nn.functional as F
from .model import AlphaZeroNet
from .move_index import num_actions
from .self_play import self_play_episode
from tqdm import tqdm
from pathlib import Path

def main(
    epochs=11,
    device=None,
    mcts_sims=25,
    lr=1e-3,
    checkpoint_dir="checkpoints",
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    n_actions = num_actions()
    model = AlphaZeroNet(num_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in tqdm(range(epochs), desc="Epochs"):
        states, pis, rewards = self_play_episode(model=model, device=device, simulations=mcts_sims)

        # convert to tensors
        states_t = torch.tensor(states, device=device, dtype=torch.float32)
        pis_t = torch.tensor(pis, device=device, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)

        model.train()
        optimizer.zero_grad()
        log_probs, values = model(states_t)
        loss_p = -torch.sum(pis_t * F.log_softmax(log_probs, dim=1)) / max(1, len(states))
        loss_v = F.mse_loss(values, rewards_t)
        loss = loss_p + loss_v
        loss.backward()
        optimizer.step()

        ckpt_path = Path(checkpoint_dir) / f"trained_model_ep{ep+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Saved] {ckpt_path} loss={loss.item():.4f}")

    # final save
    final_path = Path(checkpoint_dir) / "trained_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model saved to {final_path}")

if __name__ == "__main__":
    # run with python -m training.train
    main()
