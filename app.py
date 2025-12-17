# app.py
# ============================================================
# Compression-based Galaxy Morphology
# TWO ANALYSES: before smoothing vs after smoothing
#
# ADDED (without changing UI/HTML):
#   - Resolution sensitivity endpoint: POST /run-resolution
#   - Automatic ablation runner → CSV: POST /run-ablation
# ============================================================

from flask import Flask, jsonify, request, render_template_string, send_from_directory
import numpy as np
import time
import csv
from pathlib import Path
import io, base64

# ---------------- Backend modules ----------------
from preprocess import (
    load_images_and_labels_from_folder,
    preprocess_single_image,
    smooth_image_multiscale,
)
from compression import convert_images_to_bytes, image_to_bytes_array
from build_features import compute_features_with_anchors
from nearest_neighbors import get_k_nearest
from knn_classify import split_and_scale, train_knn, evaluate_knn
from svm_rbf_classify import train_svm_rbf, evaluate_svm

# ---------------- Plotting ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.cluster.hierarchy import linkage, dendrogram
from PIL import Image

# ============================================================
# Flask setup
# ============================================================
app = Flask(__name__)
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "fig"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ============================================================
# GLOBAL STATE (for NN galleries)
# ============================================================
STATE = {
    "pre": {"images": None, "features": None, "labels": None},
    "post": {"images": None, "features": None, "labels": None},
}

# ============================================================
# Helpers
# ============================================================
def img_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_confusion_matrix(cm, class_names, fname):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    fig.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatter(points2d, labels, fname, title=None):
    fig, ax = plt.subplots()
    ax.scatter(points2d[:, 0], points2d[:, 1], c=labels, s=10)
    if title:
        ax.set_title(title)
    fig.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_dendrogram(X, fname, max_items=50):
    n = min(len(X), max_items)
    Z = linkage(X[:n], method="average")
    fig, ax = plt.subplots(figsize=(7, 4))
    dendrogram(Z, ax=ax, no_labels=True)
    ax.set_title(f"Dendrogram (subset n={n})")
    fig.savefig(FIG_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Frontend UI (embedded)  (UNCHANGED)
# ============================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Compression-based Galaxy Morphology</title>
<style>
body {
  margin: 0;
  font-family: system-ui, sans-serif;
  background: radial-gradient(circle at top, #1f2937 0, #020617 60%);
  color: #e5e7eb;
}
.container {
  max-width: 1300px;
  margin: 40px auto;
  padding: 24px;
  background: rgba(15,23,42,.95);
  border-radius: 16px;
  border: 1px solid rgba(148,163,184,.3);
}
h1 { margin-top: 0; }
.controls { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; align-items:center; }
input, select {
  background:#020617; color:#e5e7eb;
  border-radius:999px; border:1px solid rgba(148,163,184,.4);
  padding:6px 10px;
}
button {
  border-radius:999px; padding:10px 18px; border:none;
  font-weight:700; cursor:pointer;
  background:linear-gradient(135deg,#4f46e5,#06b6d4); color:white;
}
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:16px; }
.panel {
  background:#020617; border-radius:14px; padding:14px;
  border:1px solid rgba(148,163,184,.25);
}
.panel h2 { margin: 0 0 8px; font-size: 16px; opacity:.9; }
.metric { font-size: 16px; margin: 6px 0; }
.imagesGrid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-top:12px; }
.imagesGrid img { width:100%; border-radius:10px; border:1px solid rgba(148,163,184,.25); }
.status { opacity:.85; }
.nnGrid { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
.nnCard { width: 180px; background:#0b1220; padding:8px; border-radius:12px; border:1px solid rgba(148,163,184,.25); }
.nnCard img { width:100%; border-radius:8px; }
</style>
</head>

<body>
<div class="container">
  <h1>Compression-based Galaxy Morphology</h1>

  <div class="controls">
    <label>Dataset root:
      <input id="datasetRoot" value="/Users/isomiddin/Downloads/ukidss_final" style="width:420px">
    </label>

    <label>Anchor mode:
      <select id="anchorMode">
        <option value="auto">Auto</option>
        <option value="manual">Manual</option>
      </select>
    </label>

    <label>Anchors/class:
      <input id="anchorsPerClass" type="number" value="8" style="width:90px">
    </label>

    <label>Anchor root (manual):
      <input id="anchorRoot" placeholder="/path/to/anchors" style="width:260px">
    </label>

    <label>Max/class:
      <input id="maxPerClass" type="number" placeholder="optional" style="width:110px">
    </label>

    <label>Smoothing radius (post):
      <select id="smoothRadius">
        <option value="3">3</option>
        <option value="4">4</option>
        <option value="5" selected>5</option>
        <option value="6">6</option>
        <option value="7">7</option>
        <option value="8">8</option>
        <option value="9">9</option>
      </select>
    </label>

    <button onclick="runBoth()">Run BOTH analyses</button>
  </div>

  <div class="status" id="status">Idle.</div>

  <div class="grid2" id="results" style="display:none;">
    <div class="panel">
      <h2>Before smoothing (baseline)</h2>
      <div class="metric">KNN accuracy: <b id="preKnn">—</b></div>
      <div class="metric">SVM accuracy: <b id="preSvm">—</b></div>
      <div class="imagesGrid">
        <img id="preCmKnn">
        <img id="preCmSvm">
        <img id="prePca">
        <img id="preTsne">
        <img id="preDendro">
      </div>
    </div>

    <div class="panel">
      <h2>After smoothing + threshold (test)</h2>
      <div class="metric">KNN accuracy: <b id="postKnn">—</b></div>
      <div class="metric">SVM accuracy: <b id="postSvm">—</b></div>
      <div class="imagesGrid">
        <img id="postCmKnn">
        <img id="postCmSvm">
        <img id="postPca">
        <img id="postTsne">
        <img id="postDendro">
      </div>
    </div>
  </div>

  <h2 style="margin-top:22px;">Nearest Neighbours</h2>
  <div class="controls">
    <label>Analysis:
      <select id="nnWhich">
        <option value="pre">Before</option>
        <option value="post">After</option>
      </select>
    </label>
    <label>Index:
      <input id="nnIndex" type="number" value="0" style="width:90px">
    </label>
    <label>k:
      <input id="nnK" type="number" value="5" style="width:90px">
    </label>
    <label>Metric:
      <select id="nnMetric">
        <option value="euclidean">Euclidean</option>
        <option value="cosine">Cosine</option>
      </select>
    </label>
    <button onclick="loadNN()">Load</button>
  </div>

  <div class="nnGrid" id="nnGallery"></div>

</div>

<script>
async function runBoth() {
  status.innerText = "Running both pipelines...";
  results.style.display = "none";

  const payload = {
    dataset_root: datasetRoot.value,
    anchor_mode: anchorMode.value,
    anchors_per_class: parseInt(anchorsPerClass.value),
    anchor_root: anchorRoot.value || null,
    max_per_class: maxPerClass.value ? parseInt(maxPerClass.value) : null,
    smooth_radius: parseInt(smoothRadius.value)
  };

  const r = await fetch("/run-both", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify(payload)
  });

  const d = await r.json();
  if (!r.ok) {
    status.innerText = "Error: " + (d.error || "unknown");
    return;
  }

  preKnn.innerText  = (d.pre.acc_knn * 100).toFixed(2) + "%";
  preSvm.innerText  = (d.pre.acc_svm * 100).toFixed(2) + "%";
  postKnn.innerText = (d.post.acc_knn * 100).toFixed(2) + "%";
  postSvm.innerText = (d.post.acc_svm * 100).toFixed(2) + "%";

  const t = Date.now();
  preCmKnn.src  = "/fig/" + d.pre.plots.cm_knn + "?t=" + t;
  preCmSvm.src  = "/fig/" + d.pre.plots.cm_svm + "?t=" + t;
  prePca.src    = "/fig/" + d.pre.plots.pca + "?t=" + t;
  preTsne.src   = "/fig/" + d.pre.plots.tsne + "?t=" + t;
  preDendro.src = "/fig/" + d.pre.plots.dendrogram + "?t=" + t;

  postCmKnn.src  = "/fig/" + d.post.plots.cm_knn + "?t=" + t;
  postCmSvm.src  = "/fig/" + d.post.plots.cm_svm + "?t=" + t;
  postPca.src    = "/fig/" + d.post.plots.pca + "?t=" + t;
  postTsne.src   = "/fig/" + d.post.plots.tsne + "?t=" + t;
  postDendro.src = "/fig/" + d.post.plots.dendrogram + "?t=" + t;

  results.style.display = "grid";
  status.innerText = "Done. Runtime: " + d.runtime_sec.toFixed(2) + "s";
}

async function loadNN() {
  nnGallery.innerHTML = "";
  const which = nnWhich.value;
  const idx = nnIndex.value;
  const k = nnK.value;
  const metric = nnMetric.value;

  const r = await fetch(`/nearest-neighbors?which=${which}&index=${idx}&k=${k}&metric=${metric}`);
  const d = await r.json();
  if (!r.ok) {
    status.innerText = "Error: " + (d.error || "unknown");
    return;
  }

  d.items.forEach(it => {
    const div = document.createElement("div");
    div.className = "nnCard";
    div.innerHTML = `<img src="data:image/png;base64,${it.image}"><div style="opacity:.75;font-size:12px;margin-top:6px;">idx=${it.index}</div>`;
    nnGallery.appendChild(div);
  });
}
</script>
</body>
</html>
"""

# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/fig/<path:fname>")
def fig(fname):
    return send_from_directory(FIG_DIR, fname)

@app.route("/nearest-neighbors")
def nearest_neighbors_route():
    which = request.args.get("which", "pre")
    index = int(request.args.get("index", 0))
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "euclidean")

    if which not in STATE:
        return jsonify({"error": f"unknown analysis '{which}'"}), 400

    X = STATE[which]["features"]
    imgs = STATE[which]["images"]
    if X is None or imgs is None:
        return jsonify({"error": "Run the pipelines first"}), 400

    if index < 0 or index >= len(X):
        return jsonify({"error": f"index out of range (0..{len(X)-1})"}), 400

    idxs = get_k_nearest(X, index, k, metric)
    items = [{"index": j, "image": img_to_b64(imgs[j])} for j in idxs]
    return jsonify({"items": items})


# ============================================================
# Core pipeline (added resolution)
# ============================================================
def run_one_pipeline(
    *,
    images_uint8: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    anchor_mode: str,
    anchors_per_class: int,
    anchor_root: str | None,
    tag: str,
    resolution: int = 64,
):
    """
    resolution: image-to-bytes resize size (e.g., 32/64/128)
    """

    # bytes for compression (resolution-aware)
    images_bytes = convert_images_to_bytes(images_uint8, size=(resolution, resolution), mode="L")

    # features (NCD to anchors)
    # IMPORTANT: keep anchor bytes consistent with chosen resolution
    X, anchor_info = compute_features_with_anchors(
        images_bytes=images_bytes,
        labels=labels,
        class_names=class_names,
        anchor_mode=anchor_mode,
        anchors_per_class=anchors_per_class,
        anchor_root=anchor_root,
        image_to_bytes_fn=lambda p, res=resolution: image_to_bytes_array(
            np.array(Image.open(p).convert("L")),
            size=(res, res),
            mode="L",
        ),
    )

    # store for NN gallery
    STATE[tag]["images"] = images_uint8
    STATE[tag]["features"] = X
    STATE[tag]["labels"] = labels

    # train/test
    Xtr, Xte, ytr, yte, _ = split_and_scale(X, labels)

    knn = train_knn(Xtr, ytr, k=3)
    acc_knn, cm_knn, _ = evaluate_knn(knn, Xte, yte)

    svm = train_svm_rbf(Xtr, ytr)
    acc_svm, cm_svm, _ = evaluate_svm(svm, Xte, yte)

    plots = {}

    # confusion matrices
    cm_knn_name = f"cm_knn_{tag}.png"
    cm_svm_name = f"cm_svm_{tag}.png"
    save_confusion_matrix(cm_knn, class_names, cm_knn_name)
    save_confusion_matrix(cm_svm, class_names, cm_svm_name)
    plots["cm_knn"] = cm_knn_name
    plots["cm_svm"] = cm_svm_name

    # PCA / t-SNE
    pca2 = PCA(n_components=2).fit_transform(X)
    pca_name = f"pca_{tag}.png"
    save_scatter(pca2, labels, pca_name, title=f"PCA ({tag})")
    plots["pca"] = pca_name

    tsne2 = TSNE(n_components=2, perplexity=30, init="random").fit_transform(X)
    tsne_name = f"tsne_{tag}.png"
    save_scatter(tsne2, labels, tsne_name, title=f"t-SNE ({tag})")
    plots["tsne"] = tsne_name

    # dendrogram subset
    dendro_name = f"dendrogram_{tag}.png"
    save_dendrogram(X, dendro_name, max_items=50)
    plots["dendrogram"] = dendro_name

    return {
        "acc_knn": float(acc_knn),
        "acc_svm": float(acc_svm),
        "plots": plots,
        "anchors": anchor_info,
        "resolution": int(resolution),
    }


# ============================================================
# Main UI endpoint (unchanged) — uses resolution=64 default
# ============================================================
@app.route("/run-both", methods=["POST"])
def run_both():
    t0 = time.time()
    d = request.json or {}

    dataset_root = d.get("dataset_root")
    if not dataset_root:
        return jsonify({"error": "dataset_root is required"}), 400

    anchor_mode = d.get("anchor_mode", "auto")
    anchors_per_class = int(d.get("anchors_per_class", 8))
    anchor_root = d.get("anchor_root", None)
    max_per_class = d.get("max_per_class", None)
    smooth_radius = int(d.get("smooth_radius", 5))

    if max_per_class is not None:
        max_per_class = int(max_per_class)

    # UI stays the same (fixed to 64 unless you later add UI control)
    resolution = int(d.get("resolution", 64))

    # load RAW images + REAL labels
    images_raw, labels_raw, class_names = load_images_and_labels_from_folder(
        dataset_root, mode="L", max_per_class=max_per_class
    )

    # ---------------- PRE (baseline): no smoothing, no threshold ----------------
    pre_images = images_raw.astype(np.uint8)

    # ---------------- POST (test): normalize + threshold bottom 90% + smoothing ----------------
    post_images_list = []
    for img in images_raw:
        base = preprocess_single_image(img, threshold_percent=90.0)  # uint8 after threshold
        smoothed = smooth_image_multiscale(base, scales=[smooth_radius])[0]
        post_images_list.append(smoothed)
    post_images = np.stack(post_images_list).astype(np.uint8)

    # run two pipelines
    try:
        pre = run_one_pipeline(
            images_uint8=pre_images,
            labels=labels_raw,
            class_names=class_names,
            anchor_mode=anchor_mode,
            anchors_per_class=anchors_per_class,
            anchor_root=anchor_root,
            tag="pre",
            resolution=resolution,
        )

        post = run_one_pipeline(
            images_uint8=post_images,
            labels=labels_raw,
            class_names=class_names,
            anchor_mode=anchor_mode,
            anchors_per_class=anchors_per_class,
            anchor_root=anchor_root,
            tag="post",
            resolution=resolution,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "pre": pre,
        "post": post,
        "runtime_sec": float(time.time() - t0),
    })


# ============================================================
# NEW: Resolution sensitivity endpoint (NO UI changes)
# ============================================================
@app.route("/run-resolution", methods=["POST"])
def run_resolution():
    """
    Runs the same PRE and POST pipelines at multiple resolutions.
    Returns results as JSON.
    (Call via Postman/curl; UI is unchanged.)
    """
    t0 = time.time()
    d = request.json or {}

    dataset_root = d.get("dataset_root")
    if not dataset_root:
        return jsonify({"error": "dataset_root is required"}), 400

    anchor_mode = d.get("anchor_mode", "auto")
    anchors_per_class = int(d.get("anchors_per_class", 8))
    anchor_root = d.get("anchor_root", None)
    max_per_class = d.get("max_per_class", None)
    smooth_radius = int(d.get("smooth_radius", 5))
    resolutions = d.get("resolutions", [32, 64, 128])

    if max_per_class is not None:
        max_per_class = int(max_per_class)

    images_raw, labels_raw, class_names = load_images_and_labels_from_folder(
        dataset_root, mode="L", max_per_class=max_per_class
    )

    pre_images = images_raw.astype(np.uint8)
    post_images_list = []
    for img in images_raw:
        base = preprocess_single_image(img, threshold_percent=90.0)
        smoothed = smooth_image_multiscale(base, scales=[smooth_radius])[0]
        post_images_list.append(smoothed)
    post_images = np.stack(post_images_list).astype(np.uint8)

    out = []
    for res in resolutions:
        res = int(res)
        pre = run_one_pipeline(
            images_uint8=pre_images,
            labels=labels_raw,
            class_names=class_names,
            anchor_mode=anchor_mode,
            anchors_per_class=anchors_per_class,
            anchor_root=anchor_root,
            tag=f"pre_res{res}",
            resolution=res,
        )
        post = run_one_pipeline(
            images_uint8=post_images,
            labels=labels_raw,
            class_names=class_names,
            anchor_mode=anchor_mode,
            anchors_per_class=anchors_per_class,
            anchor_root=anchor_root,
            tag=f"post_res{res}",
            resolution=res,
        )
        out.append({"resolution": res, "pre": pre, "post": post})

    return jsonify({
        "items": out,
        "runtime_sec": float(time.time() - t0),
    })


# ============================================================
# NEW: Automatic ablation runner → CSV (NO UI changes)
# ============================================================
@app.route("/run-ablation", methods=["POST"])
def run_ablation():
    """
    Runs an ablation grid and writes CSV to results/ablation_results.csv

    Grid (default):
      - resolution: [32, 64, 128]
      - threshold: [None, 90, 95, 98]
      - smoothing: [None, 3, 5, 7, 9]

    Uses:
      - PRE if threshold=None and smoothing=None
      - otherwise applies preprocess + smoothing before pipeline
    """
    d = request.json or {}

    dataset_root = d.get("dataset_root")
    if not dataset_root:
        return jsonify({"error": "dataset_root is required"}), 400

    anchor_mode = d.get("anchor_mode", "auto")
    anchors_per_class = int(d.get("anchors_per_class", 8))
    anchor_root = d.get("anchor_root", None)
    max_per_class = d.get("max_per_class", None)

    resolutions = d.get("resolutions", [32, 64, 128])
    thresholds = d.get("thresholds", [None, 90, 95, 98])
    smoothings = d.get("smoothings", [None, 3, 5, 7, 9])

    if max_per_class is not None:
        max_per_class = int(max_per_class)

    images_raw, labels_raw, class_names = load_images_and_labels_from_folder(
        dataset_root, mode="L", max_per_class=max_per_class
    )

    csv_path = RES_DIR / "ablation_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "resolution",
            "threshold_percent",
            "smoothing_radius",
            "knn_accuracy",
            "svm_accuracy",
            "notes",
        ])

        for res in resolutions:
            res = int(res)
            for thr in thresholds:
                for sm in smoothings:
                    # build processed images for this condition
                    if thr is None and sm is None:
                        imgs = images_raw.astype(np.uint8)
                        note = "raw"
                    else:
                        imgs_list = []
                        for img in images_raw:
                            x = img
                            if thr is not None:
                                x = preprocess_single_image(x, threshold_percent=float(thr))
                            else:
                                # ensure uint8 if no threshold
                                x = x.astype(np.uint8)

                            if sm is not None:
                                x = smooth_image_multiscale(x, scales=[int(sm)])[0]
                            imgs_list.append(x.astype(np.uint8))
                        imgs = np.stack(imgs_list)
                        note = "processed"

                    # run pipeline (single pass, save figures under unique tag)
                    tag = f"abl_res{res}_thr{thr}_sm{sm}".replace(".", "_")
                    result = run_one_pipeline(
                        images_uint8=imgs,
                        labels=labels_raw,
                        class_names=class_names,
                        anchor_mode=anchor_mode,
                        anchors_per_class=anchors_per_class,
                        anchor_root=anchor_root,
                        tag=tag,
                        resolution=res,
                    )

                    writer.writerow([
                        res,
                        thr,
                        sm,
                        result["acc_knn"],
                        result["acc_svm"],
                        note,
                    ])

    return jsonify({
        "status": "ok",
        "csv": str(csv_path),
        "rows": len(resolutions) * len(thresholds) * len(smoothings),
    })


# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
