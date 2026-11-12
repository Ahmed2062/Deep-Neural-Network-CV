"""
Deep Compression on an MNIST MLP, end to end:

  1. train a dense baseline
  2. iterative magnitude pruning with fine-tuning
  3. K-means weight sharing with centroid fine-tuning
  4. Huffman-code the result and write it to disk
  5. reload the file into a fresh model and re-evaluate it

Run `python main.py --help` for options.
"""

import argparse
import json
import os
import time

import torch

import config
from compression.pruning import magnitude_prune, sparsity_schedule
from compression.quantization import quantize_model
from compression.serialization import load_compressed, save_compressed, storage_report
from data.data_loader import get_data_loaders
from models.mlp import MnistMLP
from utils.metrics import count_parameters, layer_stats, sparsity, to_mb
from utils.printing import header, table
from utils.training import evaluate, fit, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--baseline-epochs", type=int, default=config.BASELINE_EPOCHS)
    p.add_argument("--sparsity", type=float, default=config.TARGET_SPARSITY)
    p.add_argument("--prune-steps", type=int, default=config.PRUNE_STEPS)
    p.add_argument("--prune-scope", choices=["global", "layer"], default=config.PRUNE_SCOPE)
    p.add_argument("--prune-finetune-epochs", type=int, default=config.PRUNE_FINETUNE_EPOCHS)
    p.add_argument("--bits", type=int, default=config.QUANT_BITS)
    p.add_argument("--quant-finetune-epochs", type=int, default=config.QUANT_FINETUNE_EPOCHS)
    p.add_argument("--quant-lr", type=float, default=config.QUANT_LR)
    p.add_argument("--dense-control", action="store_true",
                   help="also train an uncompressed model for the same total number "
                        "of epochs, to separate the effect of compression from the "
                        "effect of extra training")
    return p.parse_args()


def print_layers(model):
    table(layer_stats(model), [
        ("layer", "layer", "{}"), ("shape", "shape", "{}"),
        ("active", "active", "{:,}"), ("sparsity", "sparsity", "{:.1%}"),
        ("unique_values", "unique values", "{:,}"),
    ])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()

    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size, args.seed)
    model = MnistMLP().to(device)
    results = {"config": vars(args), "device": str(device)}

    header("DEEP COMPRESSION PIPELINE")
    print(f"device: {device}   parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------ #
    header("PHASE 1: BASELINE TRAINING")
    acc_baseline = fit(model, train_loader, test_loader, args.baseline_epochs,
                       args.lr, device, "baseline")

    # ------------------------------------------------------------------ #
    header("PHASE 2: ITERATIVE MAGNITUDE PRUNING")
    prune_log = []
    for step, target in enumerate(sparsity_schedule(args.sparsity, args.prune_steps), 1):
        magnitude_prune(model, target, scope=args.prune_scope)
        acc_before = evaluate(model, test_loader, device)
        print(f"\nstep {step}: sparsity {sparsity(model):.2%}  "
              f"accuracy right after pruning {acc_before:.2f}%")
        acc_after = fit(model, train_loader, test_loader, args.prune_finetune_epochs,
                        args.lr, device, f"prune {step}")
        prune_log.append({"sparsity": sparsity(model), "acc_before_finetune": acc_before,
                          "acc_after_finetune": acc_after})
    acc_pruned = prune_log[-1]["acc_after_finetune"] if prune_log else acc_baseline
    print()
    print_layers(model)

    # ------------------------------------------------------------------ #
    header(f"PHASE 3: K-MEANS WEIGHT SHARING ({args.bits}-bit, {2 ** args.bits} clusters)")
    kmeans = quantize_model(model, args.bits)
    acc_kmeans = evaluate(model, test_loader, device)
    print(f"accuracy right after clustering: {acc_kmeans:.2f}%")
    print("fine-tuning shared centroids (dense weights frozen):")
    acc_quant = fit(model, train_loader, test_loader, args.quant_finetune_epochs,
                    args.quant_lr, device, "centroids")
    print()
    print_layers(model)

    # ------------------------------------------------------------------ #
    header("PHASE 4: HUFFMAN CODING + SERIALIZATION")
    path = os.path.join(args.output_dir, "compressed_model.npz")
    meta = save_compressed(model, path)
    sizes = storage_report(model, meta, args.bits)
    file_bits = os.path.getsize(path) * 8
    print(f"wrote {path} ({os.path.getsize(path):,} bytes)")

    # ------------------------------------------------------------------ #
    header("VERIFICATION: DECODE FILE INTO A FRESH MODEL")
    restored = load_compressed(path, MnistMLP().to(device))
    exact = all(torch.equal(a.effective_weight(), b.effective_weight())
                for a, b in zip(
                    [m for m in model.modules() if hasattr(m, "effective_weight")],
                    [m for m in restored.modules() if hasattr(m, "effective_weight")]))
    acc_restored = evaluate(restored, test_loader, device)
    print(f"decoded weights identical to in-memory model: {exact}")
    print(f"accuracy of decoded model: {acc_restored:.2f}%")
    if not exact or abs(acc_restored - acc_quant) > 1e-9:
        raise RuntimeError("Decoded model does not match the in-memory model!")

    # ------------------------------------------------------------------ #
    if args.dense_control:
        header("CONTROL: DENSE MODEL, SAME TOTAL EPOCHS")
        total_epochs = (args.baseline_epochs + args.prune_steps * args.prune_finetune_epochs
                        + args.quant_finetune_epochs)
        set_seed(args.seed)
        control = MnistMLP().to(device)
        acc_control = fit(control, train_loader, test_loader, total_epochs,
                          args.lr, device, "dense control")
        results["dense_control"] = {"epochs": total_epochs, "accuracy": acc_control}

    # ------------------------------------------------------------------ #
    header("SUMMARY")
    base = sizes["dense_fp32"]
    rows = [
        {"stage": "Baseline (dense float32)", "acc": acc_baseline, "bits": base},
        {"stage": f"Pruned to {sparsity(model):.0%} (sparse float32)", "acc": acc_pruned,
         "bits": sizes["pruned"]},
        {"stage": f"+ {args.bits}-bit weight sharing", "acc": acc_quant, "bits": sizes["quantized"]},
        {"stage": "+ Huffman coding", "acc": acc_restored, "bits": sizes["huffman"]},
        {"stage": "Actual file on disk (.npz)", "acc": acc_restored, "bits": file_bits},
    ]
    for r in rows:
        r["kb"] = r["bits"] / 8 / 1024
        r["ratio"] = base / r["bits"]
    table(rows, [("stage", "stage", "{}"), ("acc", "test acc %", "{:.2f}"),
                 ("kb", "size KB", "{:.1f}"), ("ratio", "compression", "{:.1f}x")])
    if args.dense_control:
        print(f"\ndense control trained for {results['dense_control']['epochs']} epochs: "
              f"{results['dense_control']['accuracy']:.2f}%")
    print(f"\ntotal time: {time.time() - start:.0f}s")

    results.update({
        "accuracy": {"baseline": acc_baseline, "pruned": acc_pruned,
                     "kmeans_before_finetune": acc_kmeans, "quantized": acc_quant,
                     "decoded_from_file": acc_restored},
        "pruning_steps": prune_log,
        "kmeans": kmeans,
        "layers": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                   for r in layer_stats(model)],
        "storage_bits": {**sizes, "file_on_disk": file_bits},
        "storage_mb": {k: to_mb(v) for k, v in {**sizes, "file_on_disk": file_bits}.items()},
        "compression_ratio": {k: base / v for k, v in {**sizes, "file_on_disk": file_bits}.items()},
    })
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
