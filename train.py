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
from model import Transformer, make_src_mask, make_tgt_mask, MultiHeadAttention
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
        },
        "src_vocab": model.src_vocab,
        "tgt_vocab": model.tgt_vocab,
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
    checkpoint = torch.load(path, map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.src_vocab = checkpoint["src_vocab"]
    model.tgt_vocab = checkpoint["tgt_vocab"]
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
    import math
    import matplotlib.pyplot as plt
    from functools import partial
    from lr_scheduler import NoamScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── shared dataset setup ──────────────────────────────────────────
    train_data = Multi30kDataset(split="train")
    src_vocab  = train_data.src_vocab
    tgt_vocab  = train_data.tgt_vocab

    val_data  = Multi30kDataset(split="validation")
    val_data.set_vocab(src_vocab, tgt_vocab)
    val_data.process_data()

    test_data = Multi30kDataset(split="test")
    test_data.set_vocab(src_vocab, tgt_vocab)
    test_data.process_data()

    pad_idx = src_vocab.stoi["<pad>"]

    def make_loaders(batch_size=64):
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_idx)
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size,
            collate_fn=lambda b: collate_fn(b, pad_idx)
        )
        test_loader = DataLoader(
            test_data, batch_size=1,
            collate_fn=lambda b: collate_fn(b, pad_idx)
        )
        return train_loader, val_loader, test_loader

    def make_model(pos_encoding_type="sinusoidal"):
        m = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1,
            pos_encoding_type=pos_encoding_type,
        ).to(device)
        m.src_vocab = src_vocab
        m.tgt_vocab = tgt_vocab
        return m

    def make_optimizer(model, lr=1.0):
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
        )

    EPOCHS = 10

    # ══════════════════════════════════════════════════════════════════
    # EXP 2.1 — Noam Scheduler vs Fixed LR
    # ══════════════════════════════════════════════════════════════════
    print("\n=== EXP 2.1: Noam vs Fixed LR ===")

    for use_noam in [True, False]:
        run_name = "2.1_noam_scheduler_1" if use_noam else "2.1_fixed_lr_1e4_1"
        wandb.init(project="ASSIGNMENT 3", name=run_name, reinit=True)

        model     = make_model()
        optimizer = make_optimizer(model, lr=1.0 if use_noam else 1e-4)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000) if use_noam else None
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=0.1)
        train_loader, val_loader, _ = make_loaders()
        val_loader_single = DataLoader(
            val_data, batch_size=1,
            collate_fn=lambda b: collate_fn(b, pad_idx)
        )
        for epoch in range(EPOCHS):
            train_loss = run_epoch(train_loader, model, loss_fn,
                                   optimizer, scheduler, epoch,
                                   is_train=True, device=device)
            val_loss   = run_epoch(val_loader, model, loss_fn,
                                   None, None, epoch,
                                   is_train=False, device=device)
            
            val_bleu = evaluate_bleu(model, val_loader_single, tgt_vocab, device=device)
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_bleu":   val_bleu,
                "learning_rate": current_lr,
                "epoch":      epoch
            })
            print(f"[2.1|{run_name}] Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f} | BLEU {val_bleu:.2f}")
            
            save_checkpoint(model, optimizer, scheduler, epoch,
            path=f"/content/drive/MyDrive/da6401_assignment_3/ckpt_2.1_{run_name}_epoch{epoch}.pt")
        wandb.finish()

    # ══════════════════════════════════════════════════════════════════
    # EXP 2.2 — With vs Without Scaling Factor sqrt(1/dk)
    # ══════════════════════════════════════════════════════════════════
    print("\n=== EXP 2.2: With vs Without Scaling ===")

    for use_scale in [True, False]:
        run_name = "2.2_with_scaling" if use_scale else "2.2_without_scaling"
        wandb.init(project="ASSIGNMENT 3", name=run_name, reinit=True)

        model     = make_model()
        # patch all MHA layers to use/skip scale
        for module in model.modules():
            if isinstance(module, MultiHeadAttention):
                module.use_scale = use_scale

        optimizer = make_optimizer(model)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=0.1)
        train_loader, _, _ = make_loaders()

        # only run 1000 steps to log gradient norms
        model.train()
        step = 0
        for src, tgt in train_loader:
            if step >= 1000:
                break
            src, tgt   = src.to(device), tgt.to(device)
            tgt_in     = tgt[:, :-1]
            tgt_out    = tgt[:, 1:]
            logits     = model(src, tgt_in, make_src_mask(src), make_tgt_mask(tgt_in))
            loss       = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            grad_q = model.encoder.layers[0].self_attn.W_q.weight.grad.norm().item()
            grad_k = model.encoder.layers[0].self_attn.W_k.weight.grad.norm().item()

            wandb.log({"step": step, "grad_norm_Q": grad_q, "grad_norm_K": grad_k, "loss": loss.item()})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

        wandb.finish()

    # ══════════════════════════════════════════════════════════════════
    # EXP 2.3 — Attention Heatmaps (uses best checkpoint from 2.1 noam run)
    # ══════════════════════════════════════════════════════════════════
    print("\n=== EXP 2.3: Attention Heatmaps ===")

    wandb.init(project="ASSIGNMENT 3", name="2.3_attention_heatmaps", reinit=True)

    # load best model (noam run checkpoint)
    model = make_model()
    load_checkpoint("/content/drive/MyDrive/da6401_assignment_3/ckpt_2.1_2.1_noam_scheduler_1_epoch9.pt", model)
    model.eval()

    import spacy
    spacy_de   = spacy.load("de_core_news_sm")
    sentence   = "Ein Mann mit einem roten Hut spielt Gitarre"
    tokens     = ["<sos>"] + [t.text.lower() for t in spacy_de.tokenizer(sentence)] + ["<eos>"]
    src_ids    = [src_vocab.stoi.get(t, src_vocab.stoi["<unk>"]) for t in tokens]
    src        = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask   = make_src_mask(src)

    with torch.no_grad():
        x = model.src_embed(src) * math.sqrt(model.d_model)
        x = model.pos_encoding(x)

        # pass through all encoder layers except last
        for layer in model.encoder.layers[:-1]:
            x = layer(x, src_mask)

        # manually extract attention weights from last encoder layer
        last = model.encoder.layers[-1]
        Q = last.self_attn.split_heads(last.self_attn.W_q(x))
        K = last.self_attn.split_heads(last.self_attn.W_k(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(last.self_attn.d_k)
        scores = scores.masked_fill(src_mask, float('-inf'))
        attn_w = torch.softmax(scores, dim=-1)  # [1, num_heads, src_len, src_len]
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

    num_heads = attn_w.size(1)
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(24, 8))
    axes = axes.flatten()

    for h in range(num_heads):
        ax  = axes[h]
        w   = attn_w[0, h].cpu().numpy()
        ax.imshow(w, cmap="viridis")
        ax.set_title(f"Head {h+1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)

    plt.tight_layout()
    wandb.log({"attention_heads_last_encoder_layer": wandb.Image(fig)})
    plt.close()
    for h in range(num_heads):
      fig_h, ax_h = plt.subplots(figsize=(6, 5))
      w = attn_w[0, h].cpu().numpy()
      ax_h.imshow(w, cmap="viridis")
      ax_h.set_title(f"Head {h+1}")
      ax_h.set_xticks(range(len(tokens)))
      ax_h.set_yticks(range(len(tokens)))
      ax_h.set_xticklabels(tokens, rotation=90, fontsize=8)
      ax_h.set_yticklabels(tokens, fontsize=8)
      plt.tight_layout()
      wandb.log({f"head_{h+1}": wandb.Image(fig_h)})
    plt.close(fig_h)
    wandb.finish()

    # ══════════════════════════════════════════════════════════════════
    # EXP 2.4 — Sinusoidal PE vs Learned PE
    # ══════════════════════════════════════════════════════════════════
    print("\n=== EXP 2.4: Sinusoidal vs Learned PE ===")

    for pe_type in ["sinusoidal", "learned"]:
        run_name = f"2.4_pe_{pe_type}"
        wandb.init(project="ASSIGNMENT 3", name=run_name, reinit=True)

        model     = make_model(pos_encoding_type=pe_type)
        optimizer = make_optimizer(model)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=0.1)
        train_loader, val_loader, test_loader = make_loaders(batch_size=64)
        val_loader_single = DataLoader(val_data, batch_size=1,
                               collate_fn=lambda b: collate_fn(b, pad_idx))

        for epoch in range(EPOCHS):
            run_epoch(train_loader, model, loss_fn,
                      optimizer, scheduler, epoch,
                      is_train=True, device=device)

            bleu = evaluate_bleu(model, val_loader_single, tgt_vocab, device=device)
            wandb.log({"val_bleu": bleu, "epoch": epoch})
            print(f"[2.4|{pe_type}] Epoch {epoch} | Val BLEU {bleu:.2f}")

            save_checkpoint(model, optimizer, scheduler, epoch,
                path=f"/content/drive/MyDrive/da6401_assignment_3/ckpt_2.4_{pe_type}_epoch{epoch}.pt")

        wandb.finish()

    # ══════════════════════════════════════════════════════════════════
    # EXP 2.5 — Label Smoothing 0.1 vs 0.0
    # ══════════════════════════════════════════════════════════════════
    print("\n=== EXP 2.5: Label Smoothing ===")

    for smoothing in [0.1, 0.0]:
        run_name = f"2.5_smoothing_{smoothing}"
        wandb.init(project="ASSIGNMENT 3", name=run_name, reinit=True)

        model     = make_model()
        optimizer = make_optimizer(model)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=smoothing)
        train_loader, val_loader, _ = make_loaders()

        for epoch in range(EPOCHS):
            model.train()
            total_loss       = 0.0
            total_confidence = 0.0
            n_batches        = 0

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in   = tgt[:, :-1]
                tgt_out  = tgt[:, 1:]

                logits      = model(src, tgt_in, make_src_mask(src), make_tgt_mask(tgt_in))
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_tgt    = tgt_out.reshape(-1)

                loss = loss_fn(flat_logits, flat_tgt)

                # prediction confidence = mean softmax prob of correct token
                with torch.no_grad():
                    probs     = torch.softmax(flat_logits, dim=-1)
                    non_pad   = flat_tgt != pad_idx
                    confidence = probs[non_pad].gather(
                        1, flat_tgt[non_pad].unsqueeze(1)
                    ).squeeze(1).mean().item()
                    total_confidence += confidence

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches  += 1

            val_loss = run_epoch(val_loader, model, loss_fn,
                                 None, None, epoch,
                                 is_train=False, device=device)

            wandb.log({
                "train_loss":            total_loss / n_batches,
                "val_loss":              val_loss,
                "prediction_confidence": total_confidence / n_batches,
                "epoch":                 epoch,
            })
            save_checkpoint(
              model, optimizer, scheduler, epoch,
              path=f"/content/drive/MyDrive/da6401_assignment_3/ckpt_2.5_smooth{smoothing}_epoch{epoch}.pt"
          )
            print(f"[2.5|smoothing={smoothing}] Epoch {epoch} | "
                  f"Train {total_loss/n_batches:.4f} | "
                  f"Confidence {total_confidence/n_batches:.4f}")

        wandb.finish()



if __name__ == "__main__":
    run_training_experiment()
