"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import math
from nltk.translate.bleu_score import corpus_bleu
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():

            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(
                self.smoothing / (self.vocab_size - 1)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence
            )

            # remove pad contribution
            true_dist[:, self.pad_idx] = 0

            pad_mask = (target == self.pad_idx)

            true_dist[pad_mask] = 0

        loss = torch.sum(-true_dist * log_probs, dim=1)

        non_pad = (target != self.pad_idx)

        return loss[non_pad].mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    total_loss = 0.0

    for i, (src, tgt) in enumerate(data_iter):

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        logits = logits.reshape(-1, logits.size(-1))
        tgt_target = tgt_target.reshape(-1)

        loss = loss_fn(logits, tgt_target)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_iter)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys).to(device)

        out = model.decode(memory, src_mask, ys, tgt_mask)

        prob = out[:, -1, :]  # last token
        next_word = torch.argmax(prob, dim=-1).item()

        ys = torch.cat(
            [ys, torch.tensor([[next_word]], device=device)],
            dim=1
        )

        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt in test_dataloader:

            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src)

            pred = greedy_decode(
                model,
                src,
                src_mask,
                max_len=max_len,
                start_symbol=tgt_vocab.stoi["<sos>"],
                end_symbol=tgt_vocab.stoi["<eos>"],
                device=device,
            )

            pred_tokens = pred[0].tolist()
            tgt_tokens = tgt[0].tolist()

            # remove special tokens
            pred_tokens = [
                tgt_vocab.itos[i] for i in pred_tokens
                if i not in [tgt_vocab.stoi["<sos>"], tgt_vocab.stoi["<pad>"],tgt_vocab.stoi["<eos>"]]
            ]

            tgt_tokens = [
                tgt_vocab.itos[i] for i in tgt_tokens
                if i not in [tgt_vocab.stoi["<sos>"], tgt_vocab.stoi["<pad>"],tgt_vocab.stoi["<eos>"]]
            ]

            hypotheses.append(pred_tokens)
            references.append([tgt_tokens])

    return corpus_bleu(references, hypotheses) * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout.p,
        }
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    import wandb
    from dataset import Multi30kDataset   # assuming your dataset file
    from torch.utils.data import DataLoader
    from lr_scheduler import NoamScheduler  # if separate file

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B init
    config = {
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "lr": 1.0,
        "batch_size": 64,
        "epochs": 10,
    }

    wandb.init(project="da6401-a3", config=config)

    # Dataset + vocab
    train_data = Multi30kDataset(split="train")   # build_vocab called inside
    src_vocab  = train_data.src_vocab
    tgt_vocab  = train_data.tgt_vocab

    val_data  = Multi30kDataset(split="valid")
    val_data.set_vocab(src_vocab, tgt_vocab)
    val_data.process_data()

    test_data = Multi30kDataset(split="test")
    test_data.set_vocab(src_vocab, tgt_vocab)
    test_data.process_data()

    pad_idx = src_vocab.stoi["<pad>"]

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader   = DataLoader(val_data, batch_size=config["batch_size"],
                              collate_fn=lambda b: collate_fn(b, pad_idx))
    test_loader  = DataLoader(test_data, batch_size=1,
                              collate_fn=lambda b: collate_fn(b, pad_idx))


    

    # Model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    # Optimizer + Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"]
    )
    # Loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.stoi["<pad>"],
        smoothing=0.1
    )

    #Training loop
    for epoch in range(config["epochs"]):

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            is_train=True,
            device=device
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device
        )

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            path="checkpoint.pt"
        )

    # 7. Final BLEU evaluation
    
    bleu = evaluate_bleu(
        model,
        test_loader,
        tgt_vocab,
        device=device
    )

    print(f"\nFinal Test BLEU: {bleu:.2f}")

    wandb.log({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
