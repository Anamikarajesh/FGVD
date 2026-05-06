# FGVD Dashboard

Run from the project root:

```bash
pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

The app has two model options:

- `Our best model`: `dashboard/detection/best(m).pt` + improved classification checkpoints.
- `Paper model`: `dashboard/detection/best_yolov8n.pt` + paper-based raw SGCN checkpoints.

If you only want to test a cropped vehicle image, enable `Classify whole image as one crop` in the sidebar.

Inference uses a hierarchical cascade: L1 is predicted first, L2 is restricted to labels under that L1 parent, and L3 is restricted under the selected L2 parent. Displayed confidences remain the model's original probabilities, so a forced one-child subtree does not become a misleading 100%. The dashboard also normalizes Gabor/Sobel crop features to the same 0-1 scale used by the training feature files and uses the training Gabor setting `sigma = lambda / pi`.
