import json
import math
import os
import getpass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from Architecture import DecoderOnlyTransformer
from data import build_dataset
import sentencepiece as spm


# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return input_ids, labels


# -------------------------------------------------
# Main training
# -------------------------------------------------

def main():
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}\n")

    # ----------------------------
    # W&B login (interactive)
    # ----------------------------
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = getpass.getpass("Enter your Weights & Biases API key: ")

    wandb.init(
        project="TinyLLM",
        name=cfg["experiment_name"],
        config=cfg
    )

    # ----------------------------
    # Load tokenizer
    # ----------------------------
    sp = spm.SentencePieceProcessor()
    sp.load(cfg["tokenizer"]["model_path"])

    vocab_size = sp.get_piece_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # ----------------------------
    # Build dataset
    # ----------------------------
    dataset = build_dataset(sp)

    val_split = int(0.98 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [val_split, len(dataset) - val_split]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # ----------------------------
    # Build model
    # ----------------------------
    mcfg = cfg["model"]

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        d_ffn=mcfg["d_ffn"],
        max_len=mcfg["max_len"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")

    # ----------------------------
    # Optimizer & scheduler
    # ----------------------------
    lr = cfg["training"]["learning_rate"]
    epochs = cfg["training"]["epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.cuda.amp.GradScaler()

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_loss = float("inf")
    patience = 5
    patience_ctr = 0
    global_step = 0

    print("\n🔥 Training started\n")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for input_ids, labels in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scaler.update()
            optimizer.zero_grad()

            lr_now = lr * lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            ppl = math.exp(loss.item())

            wandb.log({
                "train/loss": loss.item(),
                "train/perplexity": ppl,
                "lr": lr_now,
                "step": global_step
            })

            pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ppl=f"{ppl:.2f}",
            lr=f"{lr_now:.2e}"
            )

            global_step += 1

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)

        wandb.log({
            "val/loss": val_loss,
            "val/perplexity": val_ppl,
            "epoch": epoch + 1
        })

        print(
            f"\n📊 Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}\n"
        )


        # ----------------------------
        # Early stopping + checkpoint
        # ----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("💾 Best model saved")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("⏹ Early stopping triggered")
                break

    wandb.finish()
    print("\n✅ Training complete")


if __name__ == "__main__":
    main()
