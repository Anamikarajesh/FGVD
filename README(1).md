# README: Graph-Based Two-Three Wheeler Classification — Improved Pipeline Design

> **Purpose of this document:** This README summarises a full technical design conversation about improving a computer vision research paper's pipeline. It is intended to bring any LLM (or human collaborator) fully up to speed on all decisions, design choices, data flows, and constraints established in this conversation — so work can continue without repeating context.

---

## 1. Source Paper

**Title:** Graph-Based Two-Three Wheeler Classification in Unconstrained Indian Roads
**Published at:** IEEE ITSC 2024 (27th International Conference on Intelligent Transportation Systems), Edmonton, Canada
**Authors:** Satyajit Nayak, Patitapaban Palo, Kwanit Gupta, Satarupa Uttarkabat
**Affiliation:** DSW, Valeo India

### Paper's Original Pipeline
The paper proposes a two-stage pipeline:
- **Stage 1:** YOLOv7-tiny for object detection
- **Stage 2:** Detected objects are represented as 2D pixel-grid graphs (64×64 = 4096 nodes) and classified using a Spatial Graph Convolutional Network (SGCN)

### Features Used (Original)
Three types of node attributes in the graph:
- RGB (colour per pixel)
- Gabor filter (texture)
- Sobel operator (edges/gradients)

### Dataset: FGVD (Fine-Grained Vehicle Detection)
- **Total images:** 5502 scene images captured from a moving camera on Indian roads (Bangalore, Hyderabad)
- **Split:** 3535 train / 884 validation / 1083 test
- **Bounding boxes:** ~24,450
- **Label hierarchy:** 3 levels
  - L-1: Vehicle type (Car, Motorcycle, Scooter, Truck, Autorickshaw, Bus) — 6 classes
  - L-2: Manufacturer (Honda, Hero, Bajaj, TVS, etc.) — 57 classes
  - L-3: Model name (Honda Activa, Hero Splendor, etc.) — 217 classes (210 unique)
- **Hierarchy depth:** Two-wheelers have a 3-level hierarchy; three-wheelers have a 2-level hierarchy

### Original Results (Best Numbers)
| Label Level | Best Feature Combo | Accuracy |
|---|---|---|
| L-1 | RGB + Gabor + Sobel | 95.05% |
| L-2 | RGB + Gabor | 71.89% |
| L-3 | RGB + Gabor | 49.87% |

---

## 2. Core Problem Identified

The large accuracy drop from L-1 → L-2 → L-3 was diagnosed as having three root causes:

1. **Shallow features:** Gabor and Sobel cannot distinguish fine-grained classes like Honda Activa vs Honda Dio — they share nearly identical textures and edges
2. **Rigid graph structure:** A 64×64 pixel grid creates 4096 noisy nodes with trivially similar neighbours — most edges carry no useful discriminative signal
3. **Uniform aggregation in SGCN:** Every neighbour gets equal weight (1/k) regardless of relevance — irrelevant background nodes pollute the representation as much as discriminative logo or panel nodes

---

## 3. Agreed Improvements

All four pipeline stages were chosen for improvement, with goals of **higher accuracy** and **better L-2/L-3 fine-grained classification**. The user requested analysis and suggestions first before implementation.

### Stage 1 — Detection: YOLOv7-tiny → Cascade R-CNN
**Reason for choice:**
- Cascade R-CNN uses progressive IoU thresholds (0.5 → 0.6 → 0.7) across three sequential detection heads
- Each head refines bounding box coordinates from the previous head, producing much tighter boxes around small vehicles
- Tighter boxes mean cleaner crops fed to Stage 2 — partial vehicle crops degrade graph features
- Better recall on small, overlapping two-wheelers in dense Indian traffic

> **Important design decision made in conversation:** The Cascade R-CNN outputs **bounding boxes only** downstream. Class scores (L-1 predictions) from the detector are **discarded** and NOT passed to any later stage. This was an explicit user decision to avoid error propagation — a wrong L-1 from the detector would permanently constrain L-2 and L-3 predictions. All classification (L-1, L-2, L-3) is handled entirely by the GAT classifier in Stage 5.

**Cascade R-CNN architecture:**
- Backbone: ResNet-50 + FPN (Feature Pyramid Network)
- FPN produces multi-scale feature maps — critical for detecting large trucks and tiny scooters in the same scene
- RPN generates ~2000 candidate proposals
- 3 cascade detection heads with IoU thresholds 0.5, 0.6, 0.7
- NMS threshold: 0.45 | Confidence threshold: 0.5
- Typical output per scene: 10–40 boxes

### Stage 2 — RoI Crop and Preprocessing
- All N detected boxes are cropped (no class filtering — since L-1 labels are discarded)
- Cropping uses **RoI Align** (bilinear interpolation), not naive pixel crop — preserves sub-pixel detail important for L-3
- Each crop resized to fixed 64×64 pixels
- Training augmentations applied: horizontal flip, colour jitter, cutout
- Output tensor shape: `[N, 3, 64, 64]` float32

### Stage 3 — Feature Extraction: Gabor/Sobel → ResNet-50 + LBP
**Reason for change:** Deep CNN features from a pretrained backbone encode semantic appearance differences that handcrafted filters cannot. A 256-dimensional feature vector from ResNet layer3 captures the difference between a Honda Activa front panel and a Honda Dio front panel — Gabor/Sobel cannot.

- **ResNet-50** pretrained on ImageNet, used up to `layer3`
- Produces a 16×16 spatial feature map with 256 channels per crop
- **Frozen initially** during first training pass (saves ~4 GB VRAM)
- Fine-tune last 2 layers in a second training pass for best accuracy
- **LBP (Local Binary Pattern)** texture histograms computed in parallel — 59-dimensional histogram
- Both concatenated and projected down to **128-dimensional node feature vector**
- Output shape: `[N, ~100, 128]` — one 128-d vector per superpixel node per crop

### Stage 4 — Graph Construction: 2D Pixel Grid → SLIC Superpixel + k-NN
**Reason for change:** The pixel grid creates 4096 nodes with trivially similar neighbours. SLIC groups perceptually similar pixel regions into ~100 meaningful superpixels, dramatically reducing graph size and making every edge semantically meaningful.

- **SLIC superpixels** applied to each 64×64 crop → ~100 nodes per crop
- Nodes connected by **k-NN (k=6)** based on combination of spatial proximity + feature vector similarity
- Edge weights encode similarity strength
- Output: PyTorch Geometric `Data` object per crop
  - `x`: `[~100, 128]` — node features
  - `edge_index`: `[2, E]` — edge connectivity
  - `edge_attr`: `[E, 1]` — edge weights

> **Performance note:** SLIC runs on CPU, not GPU. This makes Stage 4 the single biggest training bottleneck. **Pre-compute and cache all SLIC graphs to disk as `.pt` files before training begins.** This is a one-time ~2 hour job but saves 35–40% off every subsequent training epoch.

### Stage 5 — Classification: SGCN → GAT + Hierarchical Multi-Task Loss

#### GAT (Graph Attention Network) — detailed explanation

**SGCN limitation:** SGCN aggregates neighbour messages using fixed uniform weights determined purely by graph structure:
```
H' = D̂^(-1/2) · Â · D̂^(-1/2) · H · W + b
```
Every neighbour gets weight 1/k. Background and shadow nodes get the same influence as the discriminative logo node.

**GAT mechanism:** For every edge (i → j), a learnable attention score is computed:
```
e_ij  = LeakyReLU( aᵀ · [W·hᵢ ‖ W·hⱼ] )
α_ij  = softmax_j( e_ij )           ← normalised, sums to 1 across all neighbours
h'ᵢ   = σ( Σⱼ α_ij · W · hⱼ )     ← weighted aggregation
```
The key: `[W·hᵢ ‖ W·hⱼ]` concatenates features of both nodes before scoring — the weight depends on what both nodes contain, not just that an edge exists. This is **content-aware** aggregation.

**Multi-head attention:** K independent attention heads run in parallel, outputs concatenated. Each head can specialise — one may attend to shape-similar nodes, another to texture-similar nodes.

**Why this helps L-2/L-3:** In the SLIC graph, nodes correspond to vehicle parts (logo area, headlight, front panel, handlebar, wheel, background). GAT learns through training that logo and front-panel nodes consistently predict correct L-2/L-3 labels and assigns them high attention weights (e.g. 0.41). Background and shadow nodes receive near-zero attention (e.g. 0.06–0.08). SGCN cannot do this suppression.

**Expected gains from GAT:**
- L-2: +10–15 percentage points
- L-3: +12–18 percentage points

#### Three Independent Classification Heads
- **Head 1 (L-1):** Linear layer → 6 logits (vehicle type)
- **Head 2 (L-2):** Linear layer → 57 logits (manufacturer)
- **Head 3 (L-3):** Linear layer → 217 logits (model name)
- All three heads share the same GAT backbone and graph-level embedding (from global mean pooling)
- Heads are fully independent — no head sees output from another head at inference
- **No label masking** from Stage 1 (consistent with the decision to discard detector L-1 labels)

#### Hierarchical Multi-Task Loss
```
Total Loss = λ₁ · CE(pred_L1, gt_L1) + λ₂ · CE(pred_L2, gt_L2) + λ₃ · CE(pred_L3, gt_L3)
```
- Supervision comes from FGVD **ground truth labels only** — not from the detector
- Recommended weighting: λ₁=0.2, λ₂=0.3, λ₃=0.5 (higher weight on harder L-3 task)
- Forces the shared backbone to simultaneously optimise all three levels
- The original paper trained each level independently, ignoring the semantic hierarchy entirely

---

## 4. Complete Corrected Data Flow

```
INPUT
  FGVD scene image — variable resolution, RGB uint8
        ↓  resize to 800×800, normalise (ImageNet mean/std)
  Tensor: [1, 3, 800, 800] float32

STAGE 1 — Cascade R-CNN
  Backbone: ResNet-50 + FPN → multi-scale feature maps
  RPN → ~2000 region proposals
  3 cascade heads (IoU 0.5→0.6→0.7) → refined boxes
  NMS (IoU 0.45) + confidence filter (>0.5)
  OUTPUT: N bounding boxes [x1, y1, x2, y2] in absolute pixel coords
          N confidence scores
          *** L-1 class scores DISCARDED — not passed downstream ***

STAGE 2 — RoI Crop and Preprocessing
  RoI Align on all N boxes (sub-pixel bilinear crop from original image)
  Resize each crop to 64×64
  Augmentation during training: flip, colour jitter, cutout
  OUTPUT: [N, 3, 64, 64] float32

STAGE 3 — Feature Extraction
  ResNet-50 (frozen) up to layer3 → 256-d spatial features per region
  LBP texture histogram → 59-d per region
  Concat + linear projection → 128-d node feature vector
  OUTPUT: [N, ~100, 128] — one 128-d vector per superpixel per crop

STAGE 4 — Graph Construction
  SLIC superpixel segmentation → ~100 superpixel nodes per crop
  k-NN (k=6) edges by spatial proximity + feature similarity
  Edge weights = similarity strength
  OUTPUT: PyG Data object per crop
          x: [~100, 128], edge_index: [2, E], edge_attr: [E, 1]

STAGE 5 — GAT Classifier
  2× GAT layers with multi-head attention (K=8 heads)
  Global mean pooling → graph-level embedding
  3 independent linear heads:
    → L-1 logits: [N, 6]
    → L-2 logits: [N, 57]
    → L-3 logits: [N, 217]
  Loss = 0.2·CE_L1 + 0.3·CE_L2 + 0.5·CE_L3  (ground truth only)

OUTPUT
  For each detected vehicle crop:
    Predicted L-1 label (vehicle type)
    Predicted L-2 label (manufacturer)
    Predicted L-3 label (model name)
```

---

## 5. Expected Accuracy After Improvements

| Label Level | Original (best) | Expected after improvements |
|---|---|---|
| L-1 | 95.05% | ~97% |
| L-2 | 71.89% | ~83–87% |
| L-3 | 49.87% | ~65–72% |

**Priority order for implementation (highest impact first):**
1. CNN backbone features (Stage 3) — biggest single gain for L-2/L-3
2. Hierarchical multi-task loss (Stage 5) — addresses ignored label hierarchy
3. Superpixel graph (Stage 4) — reduces noise in graph structure
4. Cascade R-CNN (Stage 1) — improves box quality, smaller relative gain

---

## 6. GPU Requirements

| GPU | VRAM | Suitability |
|---|---|---|
| GTX 1650 / RTX 2060 | 4–6 GB | Original paper only — insufficient for CNN backbone |
| Colab / Kaggle T4 | 16 GB | Improved pipeline with **frozen** ResNet backbone, batch 16 |
| RTX 3090 | 24 GB | Recommended for research use, batch 32 |
| A100 / H100 | 40–80 GB | Full backbone fine-tuning, batch 64 |

**VRAM breakdown (improved pipeline):**
- YOLOv9c / Cascade R-CNN: ~2.0 GB
- ResNet-50 backbone: ~2.5 GB
- GAT + hierarchical loss (3 heads): ~3.5 GB
- SLIC graph + batch: ~2.0 GB
- Activations + overhead: ~2.0 GB
- **Total: ~10–14 GB**

**Key memory-saving strategy:** Freeze ResNet-50 during initial training (saves ~4 GB). Unfreeze last 2 layers only in second fine-tuning pass.

---

## 7. Runtime Estimates

### Training Time per Epoch (on 3535 train images)

| Stage | Colab T4 | RTX 3090 |
|---|---|---|
| S1 — Cascade R-CNN | ~18 min | ~7 min |
| S2 — RoI crop + augment | ~2 min | <1 min |
| S3 — ResNet features | ~10 min | ~4 min |
| S4 — SLIC graph construction | ~22 min | ~8 min |
| S5 — GAT + loss | ~8 min | ~3 min |
| **Total per epoch** | **~60 min** | **~23 min** |

### Total Training (50–80 epochs)

| GPU | Total time |
|---|---|
| Colab T4 (free) | 50–70 hrs |
| RTX 3090 | 18–25 hrs |
| A100 | 8–12 hrs |

### Inference on Test Set (1083 images)

| GPU | Per image | Full test set |
|---|---|---|
| Colab T4 | ~275 ms | ~5 min |
| RTX 3090 | ~103 ms | ~2 min |

---

## 8. Critical Implementation Notes

1. **SLIC is the bottleneck:** Runs on CPU only. Pre-compute and cache all graphs to `.pt` files before training. One-time cost ~2 hrs; saves 35–40% per epoch.

2. **Colab session limit:** 12 hr limit means 5–6 sessions needed for full training. Save `torch.save()` checkpoints after every epoch — not just best model.

3. **Two-phase training:**
   - Phase 1: Freeze ResNet-50 entirely. Train GAT heads only.
   - Phase 2: Unfreeze ResNet-50 last 2 layers. Fine-tune end-to-end at lower learning rate.

4. **No L-1 label passing:** Cascade R-CNN class scores must be explicitly dropped before passing boxes downstream. This is an architectural decision, not a default behaviour.

5. **All N boxes forwarded:** Do not filter crops by vehicle class after Stage 1. All detected vehicle crops (regardless of type) go through Stages 2–5. The GAT classifier handles all vehicle types.

6. **Loss weighting:** Use λ₁=0.2, λ₂=0.3, λ₃=0.5. Higher weight on L-3 pushes more gradient signal toward the hardest task.

---

## 9. Key Design Decisions Summary

| Decision | Choice Made | Reason |
|---|---|---|
| Detection model | Cascade R-CNN | Tighter boxes via progressive IoU; better small vehicle recall |
| L-1 from detector | Discarded | Prevents error propagation to L-2/L-3 |
| Class filtering after detection | None — all boxes forwarded | Avoids dependency on detector's L-1 accuracy |
| Node feature type | ResNet-50 layer3 + LBP | Semantic 256-d features vs shallow handcrafted |
| Graph type | SLIC superpixel k-NN | ~100 meaningful nodes vs 4096 noisy pixels |
| GNN architecture | GAT (Graph Attention Network) | Content-aware, learnable neighbour weighting |
| Classification heads | 3 independent heads | No cross-head dependency at inference |
| Training supervision | Ground truth labels only | Clean signal, no detector label pollution |
| Loss function | Hierarchical multi-task | Joint optimisation across all 3 label levels |

---

## 10. Files Produced in This Conversation

| File | Description |
|---|---|
| `corrected_pipeline_dataflow.svg` | Visual diagram of the full corrected 5-stage pipeline with tensor shapes and data types at each stage boundary |
| `gat_explainer.svg` | Side-by-side comparison of SGCN uniform aggregation vs GAT attention-weighted aggregation, with attention formula |
| `README.md` | This file |

---

## 11. What Has NOT Been Implemented Yet

The conversation covered analysis, design decisions, and architecture. The following implementation tasks are pending:

- [ ] Cascade R-CNN training/inference code (MMDetection or Detectron2 config)
- [ ] RoI Align crop extraction script
- [ ] ResNet-50 feature extractor with layer3 hook
- [ ] LBP texture computation per crop
- [ ] SLIC superpixel graph construction + disk caching script
- [ ] PyTorch Geometric `Data` object builder
- [ ] GAT model definition (2 layers, 8 heads, 3 output heads)
- [ ] Hierarchical multi-task loss function
- [ ] Full training loop with checkpoint saving
- [ ] Evaluation script (L-1, L-2, L-3 accuracy per class)
- [ ] Mixed precision + gradient checkpointing for Colab T4 compatibility

---

*Generated from a design conversation. All accuracy estimates are projections based on literature benchmarks for similar architectural changes; actual results will vary with hyperparameter tuning and training stability.*
