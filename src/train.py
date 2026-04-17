from __future__ import annotations

import argparse
import time
from pathlib import Path

import sacrebleu
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data import load_parallel_splits, make_dataloaders, save_tokenizers, train_tokenizers
from model import ModelConfig, TransformerMT
from utils import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    AverageMeter,
    NoamScheduler,
    count_parameters,
    ensure_dir,
    format_seconds,
    get_device,
    safe_exp,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch Transformer for machine translation.")
    parser.add_argument("--dataset", default="multi30k", choices=["multi30k", "opus100", "opus_books"])
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--src-lang", default="en")
    parser.add_argument("--tgt-lang", default="de")
    parser.add_argument("--tokenizer", default="word", choices=["word", "bpe"])
    parser.add_argument("--shared-vocab", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-type", default="mha", choices=["mha", "mqa", "gqa"])
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-aux-weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--decode-max-len", type=int, default=72)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def infer_num_kv_heads(attn_type: str, num_heads: int, requested: int | None) -> int:
    if requested is not None:
        return requested
    if attn_type == "mha":
        return num_heads
    if attn_type == "mqa":
        return 1
    if attn_type == "gqa":
        return max(1, num_heads // 2)
    raise ValueError(attn_type)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux_loss: torch.Tensor,
    label_smoothing: float,
    moe_aux_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )
    return ce + moe_aux_weight * aux_loss, ce


def train_one_epoch(
    model: TransformerMT,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        logits, aux_loss = model(src, tgt_in)
        loss, ce = compute_loss(logits, tgt_out, aux_loss, args.label_smoothing, args.moe_aux_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scheduler.step()
        optimizer.step()
        tokens = (tgt_out != PAD_ID).sum().item()
        loss_meter.update(loss.item(), tokens)
        ce_meter.update(ce.item(), tokens)
        pbar.set_postfix(loss=f"{loss_meter.avg:.3f}", ppl=f"{safe_exp(ce_meter.avg):.1f}")
    return {"loss": loss_meter.avg, "ce": ce_meter.avg, "ppl": safe_exp(ce_meter.avg)}


@torch.no_grad()
def evaluate_loss(
    model: TransformerMT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    for src, tgt in tqdm(loader, desc="valid", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits, aux_loss = model(src, tgt_in)
        loss, ce = compute_loss(logits, tgt_out, aux_loss, args.label_smoothing, args.moe_aux_weight)
        tokens = (tgt_out != PAD_ID).sum().item()
        loss_meter.update(loss.item(), tokens)
        ce_meter.update(ce.item(), tokens)
    return {"loss": loss_meter.avg, "ce": ce_meter.avg, "ppl": safe_exp(ce_meter.avg)}


@torch.no_grad()
def greedy_decode(model: TransformerMT, src: torch.Tensor, max_len: int) -> torch.Tensor:
    model.eval()
    memory, _ = model.encode(src)
    ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        dec, _ = model.decode(ys, memory, src)
        logits = model.generator(dec[:, -1])
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(EOS_ID)
        if finished.all():
            break
    return ys


@torch.no_grad()
def beam_search_decode(model: TransformerMT, src: torch.Tensor, max_len: int, beam_size: int) -> torch.Tensor:
    model.eval()
    device = src.device
    batch_outputs: list[torch.Tensor] = []
    for i in range(src.size(0)):
        src_i = src[i : i + 1]
        memory, _ = model.encode(src_i)
        beams: list[tuple[torch.Tensor, float, bool]] = [
            (torch.tensor([[BOS_ID]], device=device, dtype=torch.long), 0.0, False)
        ]
        for _ in range(max_len - 1):
            candidates: list[tuple[torch.Tensor, float, bool]] = []
            all_finished = True
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, True))
                    continue
                all_finished = False
                dec, _ = model.decode(seq, memory, src_i)
                logits = model.generator(dec[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                for log_p, idx in zip(top_log_probs[0], top_indices[0]):
                    next_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                    is_finished = idx.item() == EOS_ID
                    length_penalty = ((5 + next_seq.size(1)) / 6) ** 0.6
                    next_score = (score + log_p.item()) / length_penalty
                    candidates.append((next_seq, next_score, is_finished))
            beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
            if all_finished:
                break
        best_seq = max(beams, key=lambda item: item[1])[0].squeeze(0)
        batch_outputs.append(best_seq)
    return torch.nn.utils.rnn.pad_sequence(batch_outputs, batch_first=True, padding_value=PAD_ID)


@torch.no_grad()
def evaluate_bleu(
    model: TransformerMT,
    loader: torch.utils.data.DataLoader,
    tgt_tokenizer,
    src_tokenizer,
    device: torch.device,
    max_len: int,
    beam_size: int,
    max_batches: int | None = None,
) -> tuple[float, list[dict[str, str]]]:
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []
    for batch_idx, (src, tgt) in enumerate(tqdm(loader, desc="decode", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        src = src.to(device)
        if beam_size > 1:
            pred = beam_search_decode(model, src, max_len=max_len, beam_size=beam_size).cpu()
        else:
            pred = greedy_decode(model, src, max_len=max_len).cpu()
        for src_ids, tgt_ids, pred_ids in zip(src.cpu(), tgt, pred):
            src_text = src_tokenizer.decode(src_ids.tolist())
            ref_text = tgt_tokenizer.decode(tgt_ids.tolist())
            hyp_text = tgt_tokenizer.decode(pred_ids.tolist())
            hypotheses.append(hyp_text)
            references.append(ref_text)
            if len(samples) < 12:
                samples.append({"src": src_text, "ref": ref_text, "hyp": hyp_text})
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True).score if hypotheses else 0.0
    return bleu, samples


def write_samples(samples: list[dict[str, str]], path: Path) -> None:
    lines: list[str] = []
    for i, item in enumerate(samples, 1):
        lines.append(f"[{i}] SRC: {item['src']}")
        lines.append(f"    REF: {item['ref']}")
        lines.append(f"    HYP: {item['hyp']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return ensure_dir(args.output_dir)
    moe = "moe" if args.use_moe else "dense"
    name = f"{args.dataset}_{args.src_lang}-{args.tgt_lang}_{args.tokenizer}_{args.attn_type}_{moe}"
    return ensure_dir(Path("outputs") / name)


def main() -> dict:
    args = parse_args()
    set_seed(args.seed)
    output_dir = build_output_dir(args)
    device = torch.device("cpu") if args.cpu else get_device()
    start = time.time()

    train_examples, valid_examples, test_examples = load_parallel_splits(
        args.dataset,
        args.src_lang,
        args.tgt_lang,
        dataset_config=args.dataset_config,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    src_tok, tgt_tok = train_tokenizers(train_examples, args.tokenizer, args.vocab_size, shared_vocab=args.shared_vocab)
    save_tokenizers(src_tok, tgt_tok, output_dir, args.tokenizer)
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_examples,
        valid_examples,
        test_examples,
        src_tok,
        tgt_tok,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    num_kv_heads = infer_num_kv_heads(args.attn_type, args.num_heads, args.num_kv_heads)
    cfg = ModelConfig(
        src_vocab_size=src_tok.get_vocab_size(),
        tgt_vocab_size=tgt_tok.get_vocab_size(),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=num_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(args.max_len, args.decode_max_len) + 1,
        attn_type=args.attn_type,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_weight=args.moe_aux_weight,
    )
    model = TransformerMT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup=args.warmup)

    history: list[dict] = []
    best_valid = float("inf")
    best_path = output_dir / "best_model.pt"
    print(f"device={device} train={len(train_loader.dataset)} valid={len(valid_loader.dataset)} params={count_parameters(model):,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        valid_metrics = evaluate_loss(model, valid_loader, device, args)
        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(
            f"epoch {epoch:02d} train_loss={train_metrics['loss']:.3f} "
            f"valid_loss={valid_metrics['loss']:.3f} valid_ppl={valid_metrics['ppl']:.2f}"
        )
        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    valid_bleu, valid_samples = evaluate_bleu(
        model, valid_loader, tgt_tok, src_tok, device, max_len=args.decode_max_len, beam_size=args.beam_size, max_batches=None
    )
    test_bleu, test_samples = evaluate_bleu(
        model, test_loader, tgt_tok, src_tok, device, max_len=args.decode_max_len, beam_size=args.beam_size, max_batches=None
    )
    write_samples(test_samples or valid_samples, output_dir / "samples.txt")

    metrics = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "tokenizer": args.tokenizer,
        "attn_type": args.attn_type,
        "num_heads": args.num_heads,
        "num_kv_heads": num_kv_heads,
        "use_moe": args.use_moe,
        "num_experts": args.num_experts if args.use_moe else 0,
        "src_vocab_size": src_tok.get_vocab_size(),
        "tgt_vocab_size": tgt_tok.get_vocab_size(),
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "beam_size": args.beam_size,
        "train_size": len(train_loader.dataset),
        "valid_size": len(valid_loader.dataset),
        "test_size": len(test_loader.dataset),
        "best_valid_loss": best_valid,
        "valid_bleu": valid_bleu,
        "test_bleu": test_bleu,
        "history": history,
        "runtime": format_seconds(time.time() - start),
        "samples": test_samples or valid_samples,
        "output_dir": str(output_dir),
    }
    save_json(metrics, output_dir / "metrics.json")
    print(f"valid_bleu={valid_bleu:.2f} test_bleu={test_bleu:.2f} runtime={metrics['runtime']}")
    print(f"saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    main()
