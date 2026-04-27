FGVD Graph Handover Package — README
======================================
Paper: "Graph-based Two-Three Wheeler Classification in Unconstrained Indian Roads"
       ITSC 2024, Valeo India

This package provides the complete, modular feature set needed to reproduce the
L-1, L-2, and L-3 classification results using a Spatial Graph Convolutional
Network (SGCN).  Everything below reflects exact paper parameters.

─────────────────────────────────────────────────────────────
1. MASTER SPATIAL SKELETON   →   master_grid_skeleton.npz
─────────────────────────────────────────────────────────────
• THIS FILE IS NOW NAMED master_grid_skeleton.npz
• It stores the 8-neighbour connectivity structure ONLY.
• Weights in this file are spatial (1/dist) and should be IGNORED
  by the SGCN. Use it only to know which (u, v) pairs are neighbours.
• Per-sample edge weights must be computed at runtime using
  build_per_sample_adj(rgb_features) from build_handover.py.

─────────────────────────────────────────────────────────────
1b. PER-SAMPLE ADJACENCY MATRICES   →   per_sample_adj/{split}/
─────────────────────────────────────────────────────────────
• Correct adjacency matrices with RGB-similarity edge weights.
• Each .npz file corresponds to ONE vehicle crop.
• Filename: <vehicle_id>.npz   where vehicle_id is the index key in metadata.csv.
• Load:
    import scipy.sparse as sp
    adj = sp.load_npz("per_sample_adj/train/train_0_000.npz")  # shape (4096, 4096)

• Connectivity: 8-neighbour (same as skeleton).
• Edge weights: Gaussian similarity exp(-||RGB_i - RGB_j||² / (2σ²)) with σ=0.5
    - High weight for similar pixels, low for dissimilar.
    - Symmetrically normalised D^{-½} A D^{-½}.
• These replace the old shared master_grid_adj.npz for correct per-sample weights.

─────────────────────────────────────────────────────────────
2. NODE FEATURES   →   raw_features/{rgb,gabor,sobel}/{split}/
─────────────────────────────────────────────────────────────
Each .npy file corresponds to ONE vehicle crop.
Filename: <vehicle_id>.npy   where vehicle_id is the index key in metadata.csv.
Node ordering: ROW-MAJOR (C order), i.e. node_id = row * 64 + col.

  Feature    Shape       Dtype    Range    Details
  ─────────────────────────────────────────────────────────────────
  RGB        (4096, 3)   float32  [0, 1]   BGR→RGB, divided by 255
  Gabor      (4096, 4)   float32  [0, 1]   4 orientations (see below)
  Sobel      (4096, 1)   float32  [0, 1]   gradient magnitude |G|

RGB NORMALISATION
  Scaled to [0, 1]  (divide raw uint8 by 255).
  NOT [-1, 1] — the paper uses raw colour distances for edge weights,
  which requires a non-negative range.

GABOR PARAMETERS  (paper Section IV + Eq. 1)
  λ (wavelength)  = 6.0 px
  freq            = 1.0       (= 1/λ, confirms λ=6)
  γ (aspect ratio)= 1.0
  σ               = λ/π ≈ 1.91  (Daugman canonical relation)
  ψ (phase offset)= 0           (cosine-only response)
  θ (orientations)= 0°, 45°, 90°, 135°
  ksize           = next odd integer ≥ 6σ
  Each channel is independently min-max normalised to [0, 1].

SOBEL PARAMETERS  (paper Section III-B-1, bullet 3 + Eq. 2)
  hx = [[-1,0,1],[-2,0,2],[-1,0,1]]   3×3 horizontal kernel
  hy = [[-1,-2,-1],[0,0,0],[1,2,1]]    3×3 vertical kernel
  |G| = sqrt(Gx² + Gy²)               stored as the single feature column
  Min-max normalised to [0, 1].

─────────────────────────────────────────────────────────────
3. METADATA   →   metadata.csv
─────────────────────────────────────────────────────────────
Columns:
  vehicle_id   Unique key → used to locate .npy files
  split        train / val / test
  image_stem   Source image filename (without extension)
  obj_idx      0-based index of this object within the image
  class_name   Raw FGVD fine-grained label  (e.g. scooter_TVS_Jupiter)
  L1           Vehicle type    (motorcycle / scooter / autorickshaw / …)
  L2           Manufacturer    (Honda / Hero / TVS / Bajaj / …)
  L3           Model           (Activa / Splendor / Jupiter / … )
               For 3-wheelers (2-level hierarchy) L3 = L2.
  xmin/ymin/xmax/ymax   Bounding box in the original full image
  occluded / truncated  Flags from the XML annotation

─────────────────────────────────────────────────────────────
4. STACKING LOGIC  (which features to use per label level)
─────────────────────────────────────────────────────────────
From Table II–V in the paper:

  L-1 classification → stack RGB + Gabor + Sobel  (8 columns total: 3+4+1)
    "RGB+Gabor+Sobel achieves the best performance for L-1 classification"

  L-2 classification → stack RGB + Gabor           (7 columns total: 3+4)
    "RGB+Gabor achieves the best performance for L-2 labels"

  L-3 classification → stack RGB + Gabor           (7 columns total: 3+4)
    "RGB+Gabor achieves the best performance for L-3 labels"

Python snippet to load and stack features for one vehicle:

    import numpy as np, scipy.sparse as sp, pathlib

    root = pathlib.Path("FGVD_Graph_Handover")
    # adj = sp.load_npz(root / "master_grid_adj.npz")  # OLD: shared, wrong weights

    def load_sample(vehicle_id, split, level="L1"):
        rf = root / "raw_features"
        psa = root / "per_sample_adj"

        # Load node features
        rgb   = np.load(rf / "rgb"   / split / f"{vehicle_id}.npy")  # (4096,3)
        gabor = np.load(rf / "gabor" / split / f"{vehicle_id}.npy")  # (4096,4)
        if level == "L1":
            sobel = np.load(rf / "sobel" / split / f"{vehicle_id}.npy")  # (4096,1)
            node_features = np.concatenate([rgb, gabor, sobel], axis=1)  # (4096,8)
        else:
            node_features = np.concatenate([rgb, gabor], axis=1)         # (4096,7)

        # Load per-sample adjacency (correct RGB-similarity weights)
        adj = sp.load_npz(psa / split / f"{vehicle_id}.npz")  # (4096,4096)

        return node_features, adj

─────────────────────────────────────────────────────────────
5. SGCN ARCHITECTURE REFERENCE  (for the SGCN implementer)
─────────────────────────────────────────────────────────────
From Section IV of the paper:
  • Spatial convolutional filter:  H' = D̂^{-½} Â D̂^{-½} H W + b   (Eq. 3)
  • Graph conv filters   : 64
  • Filter size          : 3×3
  • Activation           : ReLU
  • Optimiser            : Adam,  lr = 0.001
  • Batch size           : 32
  • Dropout              : 0.5
  Reference implementation: Danel et al. "Spatial Graph Convolutional Networks"
  NeurIPS Workshop 2020, Springer LNCS.

─────────────────────────────────────────────────────────────
6. PAPER TRAIN/VAL/TEST SPLITS
─────────────────────────────────────────────────────────────
  Train : 3,535 scene images
  Val   :   884 scene images
  Test  : 1,083 scene images
  (Section IV)  These are scene-level counts; vehicle counts are higher
  because one scene can contain multiple annotated objects.
