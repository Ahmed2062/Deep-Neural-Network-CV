"""Default hyper-parameters. Every value can be overridden from the command line."""

DATA_DIR = "data"
OUTPUT_DIR = "results"
SEED = 42
BATCH_SIZE = 128
LR = 1e-3

# Phase 1: baseline
BASELINE_EPOCHS = 10

# Phase 2: iterative magnitude pruning
TARGET_SPARSITY = 0.90     # fraction of weights removed at the end
PRUNE_STEPS = 3            # prune -> fine-tune repeated this many times
PRUNE_SCOPE = "global"     # "global" threshold or same sparsity per "layer"
PRUNE_FINETUNE_EPOCHS = 3  # fine-tune epochs after each pruning step

# Phase 3: K-means weight sharing
QUANT_BITS = 5             # 2**5 = 32 shared values per layer
QUANT_FINETUNE_EPOCHS = 3  # centroid fine-tuning epochs
QUANT_LR = 1e-4
