import os
from huggingface_hub import upload_file
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")
TOKEN = os.getenv("token")

upload_file(
    path_or_fileobj=r"C:\Users\alokc\OneDrive\Documents\Code\Projects\ChessTraining\checkpoints\trained_model.pt",  # local path to your .pt file
    path_in_repo="trained_model.pt",             # filename in the Hugging Face repo
    repo_id="shivumv/chessAlphaZero",      # repo name you just created
    repo_type="model",
    token=TOKEN
)
