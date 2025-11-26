# Brain Tumor Classification — Streamlit app + Grad-CAM heatmaps

Short summary
- This project is a Streamlit-based image classifier that predicts whether an MRI contains a brain tumor, and provides Grad-CAM heatmaps that visualize which regions of the image most influenced the model's decision. I added focused heatmap support (adaptive thresholding + small-object removal) so the app highlights the tumor region instead of coloring the whole image.

Quick TL;DR — run locally (PowerShell)
1. Activate the virtual environment:

```powershell
.\myenv\Scripts\Activate.ps1
```

2. Install dependencies (first run or after updates):

```powershell
pip install -r requirements.txt
```

3. Start the Streamlit app:

```powershell
streamlit run main.py
```

4. (Optional) Generate saved heatmap examples from a script:

```powershell
python .\scripts\generate_heatmap.py            # defaults to a test image in the dataset
# or supply a custom image path
python .\scripts\generate_heatmap.py "path\to\image.jpg"
```

Where outputs are saved: `scripts/heatmaps/` (heatmap image, overlay images, focused mask, and analysis text).

 What I changed (detailed step-by-step)
 ---------------------------------------------------
 I inspected the project and then implemented fixes and features as below. Use these sections if you want to review or extend the work:
 
 1) Fix uploading/processing problems
 - File: `classifier/classify.py`
   - Updated `preprocessing()` to: convert to grayscale, resize images to model input (224×224), add channel and batch dimensions, and normalize pixel values. This fixes previous errors when uploading external MRIs that had different shapes.
 
 - The demo now includes interactive corner hotspots — click/tap any corner to trigger a playful "anti-gravity" ripple that pushes particles away (inspired by Google antigravity). Use these corner gestures to create dramatic visual responses.

2) Grad-CAM implementation and fixes
- File: `classifier/gradcam.py`
  - Implemented and hardened the Grad-CAM logic: automatic detection of last Conv2D layer, correct use of Keras inputs/outputs, and robust gradient handling.
  - Added `get_focused_mask()` to produce a binary mask that isolates the strongest activation regions using:
    - Gaussian smoothing (reduce noise)
    - Percentile-based adaptive thresholding (e.g., top 90–99%)
    - Small-object removal (drop tiny speckles)
    - A safe fallback that relaxes the threshold if no region is found
  - `analyze_heatmap()` now can use the adaptive mask and returns richer metrics: mean / max intensity, area percent, centroid, bounding box, and confidence level.
  - `overlay_heatmap_on_image()` now supports applying the colored heatmap only inside the focused mask so the rest of the image is left intact (less confusing visuals).

3) Streamlit UI integration
- File: `main.py`
  - Integrated Grad-CAM overlay and toggles into the UI
  - Added a “Highlight only strongest activation regions (focused)” checkbox and an advanced tuning expander with sliders for percentile, smoothing (sigma) and minimum area so you can control how broad/strict the mask is.
  - Reorganized tabs so users see the full overlay and focused overlay (no pure grayscale heatmap shown by default).

4) Tooling / test scripts
- Files: `scripts/generate_heatmap.py`, `scripts/test_predict_fix.py`, `scripts/test_predict.py`
  - Added/updated a script that generates heatmap + overlay + focused overlay + binary mask files, and writes a small analysis report to `scripts/heatmaps/`.
  - Provided a small `test_predict_fix.py` that ensures project root is on PYTHONPATH when running locally.

5) Dependencies
- File: `requirements.txt` was updated to include `scipy` and `matplotlib` (used in mask processing and color maps).

How the focused mask works (short)
- The heatmap values are smoothed with a small Gaussian (configurable), then a percentile-based threshold (e.g., 90–99) is used to keep the top activations. Small connected regions below a pixel area threshold are removed so the result is less noisy. The overlay image blends the original RGB image with a colored (jet) version of the heatmap only where the mask is True.

How to tune it in the app
- Slider: Percentile — higher means only the strongest activations remain (use 95–99 to get small, tight regions).
- Slider: Smoothing sigma — increase to remove speckle / noise before thresholding.
- Slider: Min area — increase to ignore tiny noisy regions and keep larger contiguous activations.

Troubleshooting & notes
- If Grad-CAM shows very broad activations across the whole image:
  - Increase Percentile and/or min_area and/or sigma to make the mask stricter.
  - If the model has learned broad global features (heatmap is everywhere), explainability methods may struggle to isolate a single tumor region — consider Grad-CAM++ / SmoothGrad / Guided Backprop or review model training / dataset.
- If you encounter missing modules, run `pip install -r requirements.txt` inside the provided virtual environment.

Next steps I can do for you (pick one):
- Add drawn bounding boxes directly onto overlay images (UI and saved files) — helpful diagnostic view.
- Add an option to download focused crops (images) and analysis report directly from the Streamlit UI.
- Try Grad-CAM++ or SmoothGrad to improve localization quality for full-image activations.

If you'd like, I can also commit small example files or a final demo notebook that demonstrates the entire flow end-to-end.

---
If you want me to continue, tell me which of the items above you'd like next (bounding-box on overlay, download focused crop, explore Grad-CAM++ / SmoothGrad). 

Embedding the interactive background into Streamlit
------------------------------------------------
I added a self-contained interactive page at `web/interactive-background.html` (full-screen canvas + JS). You can embed the HTML as the app background using Streamlit's components API. Example snippet to paste into `main.py` near the top, right after you set the page config and background utility:

```py
import streamlit.components.v1 as components

# read local file (small and self-contained) and render it full-screen
html = open('web/interactive-background.html', 'r', encoding='utf-8').read()
components.html(html, height=800, scrolling=True)
```

Tips when embedding into Streamlit:
- Set `pointer-events: none` on the canvas or container if you want the background to be non-intercepting (so UI controls remain clickable). The demo HTML already positions canvas behind UI and uses pointer-events cleverly.
- Use `components.html(..., scrolling=True)` for full-screen or `st.components.v1.html(html, height=600)` to place it in a block; you can tuck it into your layout or show it once at startup.

If you'd like I can add a direct integration into `main.py` (rendering the interactive background inside the app, tuned for light/dark) and make the canvas cooperative with Streamlit UI events (like using query strings or a local message bus). 
