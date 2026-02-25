import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


# ---------------------------- Config --------------------------------- #

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128          # shorter sequences = faster
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 1            # 1 epoch for speed; good enough for project
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    output_dir: str = "models/distilbert-agnews"   # <-- new directory


cfg = Config()


# ---------------------------- Utils ---------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    assert preds.shape == labels.shape
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}


# ------------------------ Data Preparation --------------------------- #

def load_and_prepare_data(tokenizer):
    # AG News: 4-class news classification (world, sports, business, sci/tech)
    dataset = load_dataset("ag_news")

    # Use a smaller subset of the train split to speed things up
    small_train = dataset["train"].shuffle(seed=cfg.seed).select(range(5000))
    train_valid = small_train.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = train_valid["train"]
    valid_ds = train_valid["test"]
    test_ds = dataset["test"]  # can also subset if needed

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    # Tokenize with multiple processes to use CPU cores
    train_enc = train_ds.map(tokenize_batch, batched=True, num_proc=4)
    valid_enc = valid_ds.map(tokenize_batch, batched=True, num_proc=4)
    test_enc  = test_ds.map(tokenize_batch, batched=True, num_proc=4)

    # rename 'label' -> 'labels' so model can use it directly
    train_enc = train_enc.rename_column("label", "labels")
    valid_enc = valid_enc.rename_column("label", "labels")
    test_enc  = test_enc.rename_column("label", "labels")

    # keep only relevant columns and set to torch tensors
    columns = ["input_ids", "attention_mask", "labels"]
    train_enc.set_format(type="torch", columns=columns)
    valid_enc.set_format(type="torch", columns=columns)
    test_enc.set_format(type="torch", columns=columns)

    return train_enc, valid_enc, test_enc


# ------------------------ Training Loop ------------------------------ #

def train():
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=4,   # <-- 4 classes now
    )
    model.to(device)

    train_dataset, valid_dataset, test_dataset = load_and_prepare_data(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    os.makedirs(cfg.output_dir, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.num_epochs):
        # --------------------- Train --------------------- #
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [train]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --------------------- Validate ------------------- #
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs} [val]")
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = compute_metrics(all_preds, all_labels)
        avg_val_loss = val_loss / len(valid_loader)

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} "
            f"- train_loss: {avg_train_loss:.4f}, "
            f"val_loss: {avg_val_loss:.4f}, "
            f"val_acc: {metrics['accuracy']:.4f}"
        )

        # --------------------- Save Best ------------------ #
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            save_path = os.path.join(cfg.output_dir, "best")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ New best model saved at {save_path} (val_acc={best_val_acc:.4f})")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()