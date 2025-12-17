import json
import math
import os
import getpass
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from Architecture import DecoderOnlyTransformer
from data import build_dataset
import sentencepiece as spm


# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_config(path="./config_20M.json"):
    with open(path, "r") as f:
        return json.load(f)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return input_ids, labels


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    cfg = load_config()
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}\n")

    # ----------------------------
    # W&B login
    # ----------------------------
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = getpass.getpass("Enter your Weights & Biases API key: ")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["experiment_name"],
        config=cfg
    )

    # ----------------------------
    # Tokenizer
    # ----------------------------
    sp = spm.SentencePieceProcessor()
    sp.load(cfg["tokenizer"]["model_path"])
    vocab_size = sp.get_piece_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = build_dataset(sp)

    val_size = int(0.02 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    # ----------------------------
    # Model
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
    # Optimizer & Scheduler
    # ----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"])
    )

    total_steps_est = train_cfg["epochs"] * len(train_loader)
    max_steps = train_cfg["max_steps"] or total_steps_est
    warmup_steps = int(train_cfg["warmup_ratio"] * max_steps)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"])

    # ----------------------------
    # Early stopping config
    # ----------------------------
    early_cfg = train_cfg["early_stopping"]
    eval_every_steps = train_cfg["eval_every_steps"]

    best_val_loss = float("inf")
    steps_since_improve = 0
    global_step = 0

    print("\n🔥 Training started\n")

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(train_cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for input_ids, labels in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=train_cfg["mixed_precision"]):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            lr_now = train_cfg["learning_rate"] * lr_schedule(global_step)
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
            # Validation (step-based)
            # ----------------------------
            if global_step % eval_every_steps == 0:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for vi, vl in val_loader:
                        vi = vi.to(device)
                        vl = vl.to(device)
                        logits = model(vi)
                        l = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            vl.view(-1)
                        )
                        val_loss += l.item()

                val_loss /= len(val_loader)
                val_ppl = math.exp(val_loss)

                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "step": global_step
                })

                print(
                    f"\n🔎 Step {global_step} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val PPL: {val_ppl:.2f}\n"
                )

                if val_loss < best_val_loss - early_cfg["min_delta"]:
                    best_val_loss = val_loss
                    steps_since_improve = 0
                    torch.save(model.state_dict(), "checkpoint/best_model.pt")
                    print("💾 Best model saved")
                else:
                    steps_since_improve += eval_every_steps
                    if early_cfg["enabled"] and steps_since_improve >= early_cfg["patience_steps"]:
                        print("⏹ Early stopping triggered (step-based)")
                        wandb.finish()
                        return

                model.train()

            # ----------------------------
            # Max steps stop
            # ----------------------------
            if global_step >= max_steps:
                print("⏹ Reached max_steps")
                wandb.finish()
                return

    wandb.finish()
    print("\n✅ Training complete")


if __name__ == "__main__":
    main()


