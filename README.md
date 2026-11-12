# Deep Compression for an MNIST MLP

A from-scratch PyTorch implementation of the **Deep Compression** pipeline
(Han, Mao & Dally, ICLR 2016) applied to a 4-layer MLP on MNIST:
**pruning → trained quantization (K-means weight sharing) → Huffman coding**,
followed by writing the model to disk, decoding it into a fresh network and
re-evaluating it.

**Result:** the model shrinks from 1.82 MB to 56.8 KB of encoded data
(**32.1× smaller**, 26.3× including file-container overhead) while test
accuracy goes from 97.93% to **98.37%**. The accuracy is measured on the model
decoded from the compressed file, not on the in-memory model.

---

## Results

Single run, seed 42, CPU, default settings (`python main.py --dense-control`).
Full log: [`results/run_log.txt`](results/run_log.txt) · machine-readable: [`results/results.json`](results/results.json).

| Stage | Test accuracy | Size | Compression |
|---|---|---|---|
| Baseline, dense float32 | 97.93% | 1,822.0 KB | 1.0× |
| Pruned to 90% sparsity, sparse float32 | 98.37% | 243.4 KB | 7.5× |
| + 5-bit K-means weight sharing | 98.37% | 90.5 KB | 20.1× |
| + Huffman coding | 98.37% | 56.8 KB | **32.1×** |
| Actual `.npz` file on disk | 98.37% | 69.3 KB | 26.3× |

**Control:** the same dense MLP trained for the same total number of epochs (22)
reaches 97.86%. The compressed model's higher accuracy is therefore not just a
result of extra training. Pruning appears to act as a regulariser here: the
dense model's training loss approaches zero and its test accuracy plateaus.
This is one seed, and differences of ~0.1–0.2% are within run-to-run noise.

**How each size is counted** (`deep_compression/serialization.py → storage_report`):

* **Baseline:** all 465,408 weights + 1,034 biases as float32.
* **Pruned:** a sparse matrix is not free. Each surviving weight is stored
  as float32 *plus* a fixed-width relative index (the gap to the previous
  surviving weight), plus float32 biases.
* **Quantized:** each surviving weight becomes a 5-bit cluster index, plus
  the same relative index, plus a 32-entry float32 codebook per layer.
* **Huffman:** both index streams are Huffman-coded. The code tables, codebooks
  and biases are included in the total.
* **File on disk:** the above inside an uncompressed `.npz`. The extra
  ~12 KB is container overhead (zip and `.npy` headers for 44 small arrays),
  not model data. No zlib/DEFLATE is used, so the size reflects this
  project's own encoding.

### Per-layer view after compression

| Layer | Shape | Surviving weights | Sparsity | Distinct values |
|---|---|---|---|---|
| `classifier.0` | 256 × 784 | 16,503 | 91.8% | 32 |
| `classifier.2` | 512 × 256 | 18,818 | 85.6% | 32 |
| `classifier.4` | 256 × 512 | 10,640 | 91.9% | 32 |
| `classifier.6` | 10 × 256 | 580 | 77.3% | 16 |

Global pruning removes most from the input layer (many MNIST border pixels
carry no information) and least from the output layer.

### Where the bits go

| Component | Bytes |
|---|---|
| Huffman-coded cluster indices (≈4.4 bits/weight, down from 5) | 25,610 |
| Huffman-coded position gaps (≈4.2 bits/weight) | 25,036 |
| Biases (float32) | 4,136 |
| Huffman tables | 2,880 |
| Codebooks (float32) | 512 |

Storing *where* the surviving weights are costs as much as storing their values.
Huffman coding helps the position stream most, because small gaps are far
more common than large ones.

---

## Pipeline

```
 dense MLP ──► train ──► prune ─► fine-tune ─┐  ×3 (54% → 78% → 90%)
                            ▲────────────────┘
          ──► K-means (32 clusters/layer) ──► fine-tune centroids
          ──► Huffman-code indices + gaps ──► write .npz
          ──► decode into fresh model ──► verify identical weights + accuracy
```

### 1. Baseline training
MLP `784 → 256 → 512 → 256 → 10` with ReLU (466,442 parameters), Adam
(lr 1e-3), 10 epochs, normalised inputs.

### 2. Iterative magnitude pruning
Weights with the smallest magnitude are removed through a binary mask that is
multiplied into the weight on every forward pass (`deep_compression/linear.py`).
Masked weights receive zero gradient, so they stay pruned during fine-tuning.

* **Global threshold** across all layers, so each layer's sparsity is decided
  by the data rather than set by hand.
* **Iterative schedule:** prune to 54%, fine-tune, then 78%, then 90%. Each
  step removes the same fraction of the *remaining* weights. Accuracy right
  after the jump to 90% drops to 91%, and 3 epochs of fine-tuning recover it.
* **Exact sparsity:** the k smallest weights are selected by index, not by a
  `> threshold` comparison, so tied values cannot over-prune.
* Each training phase uses a **fresh optimizer**, so Adam's momentum from the
  previous phase cannot move pruned weights.

### 3. Trained quantization (K-means weight sharing)
Each layer's surviving weights are clustered into 2⁵ = 32 values with 1-D
K-means. Centroids are **initialised linearly** between the min and max weight,
as the paper recommends, so the rare large weights keep good representatives.
The layer then stores a `codebook` and an `indices` tensor, and its effective
weight is `codebook[indices] * mask`.

The codebook is an `nn.Parameter`, so fine-tuning it with backprop gives each
centroid the **sum of the gradients of all weights in its cluster**, which is
the paper's centroid update, without any custom gradient code.

### 4. Huffman coding and serialization
Two integer streams per layer are entropy-coded with a canonical Huffman code
(`deep_compression/huffman.py`): the cluster index of each surviving weight, and
the gap between consecutive surviving weights. Canonical codes mean only each
symbol's code *length* has to be stored.

### 5. Verification
`load_compressed` rebuilds a brand-new model from the file alone. The script
checks that every decoded weight is bit-identical to the in-memory model and
that the decoded model's accuracy matches. It stops with an error if either
check fails. Huffman coding is lossless, so accuracy must not change at
this step.

---

## Project structure

```
├── main.py                    # runs the full pipeline, CLI options
├── config.py                  # default hyper-parameters
├── requirements.txt
├── deep_compression/
│   ├── linear.py              # MaskedLinear: mask + codebook/indices
│   ├── pruning.py             # global / per-layer magnitude pruning, schedule
│   ├── quantization.py        # 1-D K-means weight sharing
│   ├── huffman.py             # canonical Huffman encoder/decoder
│   └── serialization.py       # file format, decoding, storage accounting
├── models/mlp.py              # MnistMLP
├── data/data_loader.py        # MNIST loaders (dataset downloaded on first run)
├── utils/                     # training loop, metrics, printing
├── tests/                     # unit tests (Huffman, pruning, K-means, round trip)
├── results/                   # run log and results.json from the run above
└── reports/                   # course project reports (PDF)
```

---

## Usage

```bash
pip install -r requirements.txt
python main.py                      # full pipeline, ~3–4 min on a laptop CPU
python main.py --dense-control      # also train the same-epochs dense control
python main.py --sparsity 0.95 --bits 4 --prune-scope layer   # experiment
python -m pytest -q                 # run the tests
```

MNIST is downloaded to `data/` on the first run. Results are written to
`results/`. Run `python main.py --help` for all options.

---

## Limitations and next steps

* **One seed, one dataset.** Results should be averaged over several seeds
  before small accuracy differences are read as meaningful.
* **Storage, not speed.** Unstructured sparsity shrinks the file but does not
  speed up inference on standard dense hardware. That would need sparse
  kernels, or structured pruning that removes whole neurons.
* **Decoded model runs in float32.** The compressed format is for storage and
  transfer; weights are expanded back to float32 for inference.
* **Biases are kept in float32.** That is 4 KB, about 7% of the payload.
* Possible extensions: per-layer bit widths, a sensitivity analysis of
  accuracy vs. sparsity per layer, and a convolutional network.

---

## Corrections to the earlier version of this repository

An earlier version of this README reported a 31.3× compression ratio. That
figure came from a bug: the serializer cast float weights to `int8`, which
truncated every weight (all smaller than 1 in magnitude) to zero. The
"compressed" model was all zeros, and its accuracy was never measured. That
version also described K-means quantization, Huffman coding and global pruning
that were not in the code. All of these are now implemented and verified as
described above. The final report in `reports/` was written for the earlier
version, so its compression figures are superseded by this README.

---

## Reference

Song Han, Huizi Mao, William J. Dally. *Deep Compression: Compressing Deep
Neural Networks with Pruning, Trained Quantization and Huffman Coding.*
ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
