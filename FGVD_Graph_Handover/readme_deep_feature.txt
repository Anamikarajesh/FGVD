============================================================
HANDOVER NOTE FOR SGCN PERSON
============================================================
One set of deep features works for ALL label levels (L1, L2, L3).

    import numpy as np, scipy.sparse as sp
    from pathlib import Path

    root = Path("/home/cse/Desktop/btech_project/FGVD_Graph_Handover")
    adj  = sp.load_npz(root / "master_grid_adj.npz")  # unchanged

    # Load deep features  (shape: 4096 × 64)
    vehicle_id = "train_00042_003"
    node_feats = np.load(
        root / "deep_features" / "multilevel" / "train" / f"{vehicle_id}.npy"
    )  # shape (4096, 64)

    # Use the SAME node_feats for L1, L2, and L3 SGCN runs.
    # Only the label file changes between levels.
============================================================