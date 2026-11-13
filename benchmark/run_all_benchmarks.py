"""
Head-to-head benchmark runner comparing qore, zxing-cpp, wechat-qr, and opencv-qr
across both python-qrcode-synthetic-benchmark (780 images) and BoofCV QR v3
(26 decoding images + 536 detection images / 1,232 GT quads).

Outputs:
  - benchmark/results/comparison_data.json
  - benchmark/results/comparison_summary.md
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import boofcv_eval as be

REPO_ROOT = os.path.dirname(BASE_DIR)
QORE_BIN = os.path.join(REPO_ROOT, "bin", "qore")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "synthetic", "test_images")
BOOFCV_DIR = be.BOOFCV_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ENGINE_ORDER = ["qore", "zxing-cpp", "wechat-qr", "opencv-qr"]

SYNTHETIC_CAT_LABELS = {
    "A": "A (Placement & Scale)",
    "B": "B (Lighting, Contrast & Noise)",
    "C": "C (Perspective, Distortion & Rotation)",
    "D": "D (Camera Capture, Blur & Resolution)",
    "E": "E (Combined Hard Scenarios)",
    "F": "F (Multi-QR Scenes)",
}


class QoreSubprocessReader:
    """Persistent `qore --batch-stdin --raw --json --multi` subprocess reader."""

    name = "qore"

    def __init__(self, unreadable: bool = False) -> None:
        self.unreadable = unreadable
        self._proc: subprocess.Popen[str] | None = None
        self.wall_total_ms: float = 0.0

    def __enter__(self) -> "QoreSubprocessReader":
        if not os.path.exists(QORE_BIN):
            raise FileNotFoundError(f"qore binary not found: {QORE_BIN}")
        cmd = [QORE_BIN, "--batch-stdin", "--raw", "--json", "--multi"]
        if self.unreadable:
            cmd.append("--unreadable")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._proc is not None:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
            self._proc = None

    def predict_path(self, image_path: str) -> tuple[be.Prediction, float]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        t0 = time.perf_counter()
        self._proc.stdin.write(image_path + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        self.wall_total_ms += (time.perf_counter() - t0) * 1000.0
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"qore subprocess closed unexpectedly on {image_path}: {err}")

        payload = json.loads(line)
        if payload.get("error"):
            raise RuntimeError(f"qore error on {image_path}: {payload['error']}")

        elapsed_ms = float(payload.get("elapsed_ms", 0.0))
        texts: list[str] = []
        quads: list[np.ndarray] = []
        for item in payload.get("results") or []:
            if item.get("decoded") and item.get("text"):
                texts.append(item["text"])
            pts = item.get("points") or []
            if len(pts) >= 4:
                quad = np.array([[p["x"], p["y"]] for p in pts[:4]], dtype=np.float32)
                quads.append(quad)
        return be.Prediction(texts=texts, quads=quads), elapsed_ms


def load_raw_bgr(image_path: str) -> np.ndarray:
    """Load BGR image ignoring EXIF orientation (matches BoofCV GT & qore --raw)."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        # Fallback for any non-ASCII path edge cases
        buf = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return img


def build_python_readers() -> dict[str, be.ReaderBase]:
    readers: dict[str, be.ReaderBase] = {}
    for cls in (be.ZxingCppReader, be.WeChatQRReader, be.OpenCVQRReader):
        r = cls()
        if r.setup():
            readers[r.name] = r
    return readers


def evaluate_synthetic(py_readers: dict[str, be.ReaderBase]) -> dict[str, Any]:
    manifest_path = os.path.join(SYNTHETIC_DIR, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    total_images = len(manifest)
    print(f"\n=== Suite 1: Synthetic Benchmark ({total_images} images) ===")

    per_engine: dict[str, dict[str, Any]] = {}

    def run_engine(name: str, decode_fn: Any) -> dict[str, Any]:
        print(f"  Running {name} on {total_images} synthetic images...")
        correct_count = 0
        total_ms = 0.0
        cat_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0, "total_ms": 0.0})
        scen_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0, "total_ms": 0.0})
        data_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0})

        for entry in manifest:
            img_path = os.path.join(SYNTHETIC_DIR, entry["filename"])
            expected = entry["expected"]
            if isinstance(expected, str):
                expected = [expected]

            decoded_texts, elapsed_ms = decode_fn(img_path)
            total_ms += elapsed_ms

            found = sum(1 for exp in expected if exp in decoded_texts)
            ok = found == len(expected) and len(expected) > 0
            if ok:
                correct_count += 1

            scen = entry["scenario"]
            cat = scen.split("_", 1)[0][:1]
            dname = entry["data_name"]

            cat_stats[cat]["total"] += 1
            cat_stats[cat]["total_ms"] += elapsed_ms
            scen_stats[scen]["total"] += 1
            scen_stats[scen]["total_ms"] += elapsed_ms
            data_stats[dname]["total"] += 1
            if ok:
                cat_stats[cat]["correct"] += 1
                scen_stats[scen]["correct"] += 1
                data_stats[dname]["correct"] += 1

        categories_out = {}
        for c in sorted(cat_stats):
            st = cat_stats[c]
            n = st["total"]
            categories_out[c] = {
                "correct": st["correct"],
                "total": n,
                "accuracy_pct": round(st["correct"] / n * 100.0, 2) if n else 0.0,
                "total_ms": round(st["total_ms"], 2),
                "avg_ms": round(st["total_ms"] / n, 2) if n else 0.0,
            }

        scenarios_out = {}
        for s in sorted(scen_stats):
            st = scen_stats[s]
            n = st["total"]
            scenarios_out[s] = {
                "correct": st["correct"],
                "total": n,
                "accuracy_pct": round(st["correct"] / n * 100.0, 2) if n else 0.0,
                "avg_ms": round(st["total_ms"] / n, 2) if n else 0.0,
            }

        data_out = {}
        for d in sorted(data_stats):
            st = data_stats[d]
            n = st["total"]
            data_out[d] = {
                "correct": st["correct"],
                "total": n,
                "accuracy_pct": round(st["correct"] / n * 100.0, 2) if n else 0.0,
            }

        summary = {
            "correct": correct_count,
            "total": total_images,
            "accuracy_pct": round(correct_count / total_images * 100.0, 2) if total_images else 0.0,
            "total_ms": round(total_ms, 2),
            "avg_ms": round(total_ms / total_images, 2) if total_images else 0.0,
            "categories": categories_out,
            "scenarios": scenarios_out,
            "data_names": data_out,
        }
        print(
            f"    -> {correct_count}/{total_images} ({summary['accuracy_pct']:.2f}%), "
            f"total={summary['total_ms']:.1f}ms, avg={summary['avg_ms']:.2f}ms/img"
        )
        return summary

    with QoreSubprocessReader(unreadable=False) as qore:
        def qore_decode(path: str) -> tuple[list[str], float]:
            pred, ms = qore.predict_path(path)
            return pred.texts, ms

        per_engine["qore"] = run_engine("qore", qore_decode)
        per_engine["qore"]["wall_roundtrip_total_ms"] = round(qore.wall_total_ms, 2)

    for name in ("zxing-cpp", "wechat-qr", "opencv-qr"):
        reader = py_readers[name]

        def py_decode(path: str, r: be.ReaderBase = reader) -> tuple[list[str], float]:
            img = load_raw_bgr(path)
            t0 = time.perf_counter()
            try:
                pred = r.predict(img)
            except Exception:
                pred = be.Prediction(texts=[], quads=[])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return pred.texts, elapsed_ms

        per_engine[name] = run_engine(name, py_decode)

    return {
        "total_images": total_images,
        "categories": sorted({e["scenario"].split("_", 1)[0][:1] for e in manifest}),
        "scenarios": sorted({e["scenario"] for e in manifest}),
        "engines": per_engine,
    }


def evaluate_boofcv_decoding(py_readers: dict[str, be.ReaderBase]) -> dict[str, Any]:
    decoding_dir = os.path.join(BOOFCV_DIR, "decoding")
    image_paths = be.list_images(decoding_dir)
    dataset: list[tuple[str, str, str, str]] = []
    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            gt_raw = f.read()
        dataset.append((os.path.basename(img_path), img_path, gt_raw, be.normalize_text(gt_raw)))

    n = len(dataset)
    print(f"\n=== Suite 2A: BoofCV Decoding ({n} images) ===")
    per_engine: dict[str, dict[str, Any]] = {}

    def run_engine(name: str, predict_fn: Any) -> dict[str, Any]:
        correct_norm = 0
        correct_raw = 0
        total_ms = 0.0
        for _, img_path, gt_raw, gt_norm in dataset:
            pred, elapsed_ms = predict_fn(img_path)
            total_ms += elapsed_ms
            if any(t == gt_raw for t in pred.texts):
                correct_raw += 1
            if any(be.normalize_text(t) == gt_norm for t in pred.texts):
                correct_norm += 1
        res = {
            "total_images": n,
            "correct_norm": correct_norm,
            "accuracy_norm_pct": round(correct_norm / n * 100.0, 2) if n else 0.0,
            "correct_raw": correct_raw,
            "accuracy_raw_pct": round(correct_raw / n * 100.0, 2) if n else 0.0,
            "total_ms": round(total_ms, 2),
            "avg_ms": round(total_ms / n, 2) if n else 0.0,
        }
        print(
            f"  {name:10s}: norm={correct_norm}/{n} ({res['accuracy_norm_pct']:.1f}%), "
            f"raw={correct_raw}/{n}, avg={res['avg_ms']:.2f}ms/img"
        )
        return res

    with QoreSubprocessReader(unreadable=False) as qore:
        per_engine["qore"] = run_engine("qore", qore.predict_path)
        per_engine["qore"]["wall_roundtrip_total_ms"] = round(qore.wall_total_ms, 2)

    for name in ("zxing-cpp", "wechat-qr", "opencv-qr"):
        reader = py_readers[name]

        def py_predict(path: str, r: be.ReaderBase = reader) -> tuple[be.Prediction, float]:
            img = load_raw_bgr(path)
            t0 = time.perf_counter()
            try:
                pred = r.predict(img)
            except Exception:
                pred = be.Prediction(texts=[], quads=[])
            return pred, (time.perf_counter() - t0) * 1000.0

        per_engine[name] = run_engine(name, py_predict)

    return {"total_images": n, "engines": per_engine}


def evaluate_boofcv_detection(py_readers: dict[str, be.ReaderBase]) -> dict[str, Any]:
    detection_root = os.path.join(BOOFCV_DIR, "detection")
    categories = sorted(
        d for d in os.listdir(detection_root) if os.path.isdir(os.path.join(detection_root, d))
    )

    cat_items: dict[str, list[tuple[str, list[np.ndarray]]]] = {}
    total_images = 0
    total_gt_quads = 0
    for cat in categories:
        cat_dir = os.path.join(detection_root, cat)
        items = []
        for img_path in be.list_images(cat_dir):
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            gt_quads = be.parse_gt_quads(txt_path)
            items.append((img_path, gt_quads))
            total_images += 1
            total_gt_quads += len(gt_quads)
        cat_items[cat] = items

    print(f"\n=== Suite 2B: BoofCV Detection ({len(categories)} categories, {total_images} images, {total_gt_quads} GT quads) ===")
    per_engine: dict[str, dict[str, Any]] = {}

    def run_engine(name: str, predict_fn: Any) -> dict[str, Any]:
        print(f"  Running {name}...")
        by_category: dict[str, dict[str, Any]] = {}
        overall_decoded = 0
        overall_detected = 0
        overall_ms = 0.0

        for cat in categories:
            items = cat_items[cat]
            cat_gt = sum(len(gt) for _, gt in items)
            cat_decoded = 0
            cat_detected = 0
            cat_ms = 0.0

            for img_path, gt_quads in items:
                pred, elapsed_ms = predict_fn(img_path)
                cat_ms += elapsed_ms
                cat_decoded += min(len(pred.texts), len(gt_quads))
                cat_detected += be.match_quads_greedy(pred.quads, gt_quads)

            n_imgs = len(items)
            overall_decoded += cat_decoded
            overall_detected += cat_detected
            overall_ms += cat_ms

            by_category[cat] = {
                "images": n_imgs,
                "gt_quads": cat_gt,
                "decoded_qr": cat_decoded,
                "decode_rate_pct": round(cat_decoded / cat_gt * 100.0, 2) if cat_gt else 0.0,
                "detected_quads": cat_detected,
                "detection_rate_pct": round(cat_detected / cat_gt * 100.0, 2) if cat_gt else 0.0,
                "total_ms": round(cat_ms, 2),
                "avg_ms": round(cat_ms / n_imgs, 2) if n_imgs else 0.0,
            }

        res = {
            "total_images": total_images,
            "total_gt_quads": total_gt_quads,
            "decoded_qr": overall_decoded,
            "decode_rate_pct": round(overall_decoded / total_gt_quads * 100.0, 2) if total_gt_quads else 0.0,
            "detected_quads": overall_detected,
            "detection_rate_pct": round(overall_detected / total_gt_quads * 100.0, 2) if total_gt_quads else 0.0,
            "total_ms": round(overall_ms, 2),
            "avg_ms": round(overall_ms / total_images, 2) if total_images else 0.0,
            "categories": by_category,
        }
        print(
            f"    -> decoded={overall_decoded}/{total_gt_quads} ({res['decode_rate_pct']:.1f}%), "
            f"detected={overall_detected}/{total_gt_quads} ({res['detection_rate_pct']:.1f}%), "
            f"avg={res['avg_ms']:.2f}ms/img"
        )
        return res

    with QoreSubprocessReader(unreadable=True) as qore:
        per_engine["qore"] = run_engine("qore", qore.predict_path)
        per_engine["qore"]["wall_roundtrip_total_ms"] = round(qore.wall_total_ms, 2)

    for name in ("zxing-cpp", "wechat-qr", "opencv-qr"):
        reader = py_readers[name]

        def py_predict(path: str, r: be.ReaderBase = reader) -> tuple[be.Prediction, float]:
            img = load_raw_bgr(path)
            t0 = time.perf_counter()
            try:
                pred = r.predict(img)
            except Exception:
                pred = be.Prediction(texts=[], quads=[])
            return pred, (time.perf_counter() - t0) * 1000.0

        per_engine[name] = run_engine(name, py_predict)

    return {
        "categories": categories,
        "total_images": total_images,
        "total_gt_quads": total_gt_quads,
        "engines": per_engine,
    }


def _md_table(headers: list[str], rows: list[list[str]], align_right_cols: set[int] | None = None) -> str:
    align_right_cols = align_right_cols or set()
    sep = [
        "---:" if idx in align_right_cols else "---"
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown_summary(data: dict[str, Any]) -> str:
    syn = data["synthetic"]
    dec = data["boofcv_decoding"]
    det = data["boofcv_detection"]
    engines = ENGINE_ORDER

    lines: list[str] = []
    lines.append("# QR Code Benchmark Comparison Summary")
    lines.append("")
    lines.append(
        "Head-to-head evaluation of **`qore`**, **`zxing-cpp`**, **`wechat-qr`**, and **`opencv-qr`** "
        f"on `{platform.machine()}` ({platform.platform()})."
    )
    lines.append("")
    lines.append(
        "> **Timing methodology**: For all 4 engines, timings measure pure QR detection & decoding on the "
        "in-memory image (excluding disk I/O and PNG/JPEG decoding). `qore` runs via a persistent "
        "`--batch-stdin --raw --json --multi` subprocess (`--unreadable` enabled for BoofCV detection so "
        "unreadable detected quads are counted symmetrically with `wechat-qr` / `opencv-qr`). "
        "`--raw` and `cv2.IMREAD_IGNORE_ORIENTATION` ensure raw pixel coordinates align with BoofCV ground truth."
    )
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    exec_headers = [
        "Engine",
        f"Synthetic ({syn['total_images']} imgs)",
        "Synthetic ms/img",
        f"BoofCV Dec ({dec['total_images']} imgs)",
        "BoofCV Dec ms/img",
        f"BoofCV Det Decoded ({det['total_gt_quads']} GT)",
        f"BoofCV Det Detected ({det['total_gt_quads']} GT)",
        "BoofCV Det ms/img",
    ]
    exec_rows = []
    for eng in engines:
        se = syn["engines"][eng]
        de = dec["engines"][eng]
        te = det["engines"][eng]
        exec_rows.append([
            f"**{eng}**" if eng == "qore" else eng,
            f"{se['correct']}/{se['total']} ({se['accuracy_pct']:.2f}%)",
            f"{se['avg_ms']:.2f}",
            f"{de['correct_norm']}/{de['total_images']} ({de['accuracy_norm_pct']:.1f}%)",
            f"{de['avg_ms']:.2f}",
            f"{te['decoded_qr']}/{te['total_gt_quads']} ({te['decode_rate_pct']:.1f}%)",
            f"{te['detected_quads']}/{te['total_gt_quads']} ({te['detection_rate_pct']:.1f}%)",
            f"{te['avg_ms']:.2f}",
        ])
    lines.append(_md_table(exec_headers, exec_rows, {1, 2, 3, 4, 5, 6, 7}))
    lines.append("")

    # 2. Synthetic Category Breakdown
    lines.append("## 2. Synthetic Benchmark — Category Breakdown (A–F)")
    lines.append("")
    cat_headers = ["Category"] + [f"{e} (Acc / Avg ms)" for e in engines]
    cat_rows = []
    for cat in syn["categories"]:
        row = [SYNTHETIC_CAT_LABELS.get(cat, cat)]
        for eng in engines:
            c = syn["engines"][eng]["categories"][cat]
            row.append(f"{c['correct']}/{c['total']} ({c['accuracy_pct']:.1f}%) / {c['avg_ms']:.1f} ms")
        cat_rows.append(row)
    total_row = ["**Total**"]
    for eng in engines:
        se = syn["engines"][eng]
        total_row.append(f"**{se['correct']}/{se['total']} ({se['accuracy_pct']:.2f}%)** / **{se['avg_ms']:.2f} ms**")
    cat_rows.append(total_row)
    lines.append(_md_table(cat_headers, cat_rows, {1, 2, 3, 4}))
    lines.append("")

    # 3. Synthetic Scenario Breakdown
    lines.append("## 3. Synthetic Benchmark — All 65 Scenarios (A01–F05)")
    lines.append("")
    scen_headers = ["Scenario"] + engines
    scen_rows = []
    for scen in syn["scenarios"]:
        row = [f"`{scen}`"]
        for eng in engines:
            s = syn["engines"][eng]["scenarios"][scen]
            row.append(f"{s['correct']}/{s['total']} ({s['accuracy_pct']:.0f}%)")
        scen_rows.append(row)
    lines.append(_md_table(scen_headers, scen_rows, {1, 2, 3, 4}))
    lines.append("")

    # 4. BoofCV Decoding
    lines.append("## 4. BoofCV QR v3 — Decoding Benchmark (26 images)")
    lines.append("")
    dec_headers = ["Engine", "Normalized Match", "Accuracy % (Norm)", "Raw Exact Match", "Total ms", "Avg ms/img"]
    dec_rows = []
    for eng in engines:
        de = dec["engines"][eng]
        dec_rows.append([
            f"**{eng}**" if eng == "qore" else eng,
            f"{de['correct_norm']}/{de['total_images']}",
            f"{de['accuracy_norm_pct']:.1f}%",
            f"{de['correct_raw']}/{de['total_images']}",
            f"{de['total_ms']:.2f}",
            f"{de['avg_ms']:.2f}",
        ])
    lines.append(_md_table(dec_headers, dec_rows, {1, 2, 3, 4, 5}))
    lines.append("")

    # 5. BoofCV Detection 16-Category Breakdown
    lines.append("## 5. BoofCV QR v3 — Detection + Decoding by Category (16 categories)")
    lines.append("")
    lines.append(
        "Quad matching uses `boofcv_eval.py`'s 1-to-1 greedy matcher (`IoU >= 0.3` OR mutual centers inside each other's quad)."
    )
    lines.append("")
    det_headers = ["Category", "Imgs", "GT"]
    for eng in engines:
        det_headers.append(f"{eng} Decoded")
        det_headers.append(f"{eng} Detected")
        det_headers.append(f"{eng} ms/img")

    det_rows = []
    for cat in det["categories"]:
        first = det["engines"][engines[0]]["categories"][cat]
        row = [f"`{cat}`", str(first["images"]), str(first["gt_quads"])]
        for eng in engines:
            c = det["engines"][eng]["categories"][cat]
            row.append(f"{c['decoded_qr']} ({c['decode_rate_pct']:.1f}%)")
            row.append(f"{c['detected_quads']} ({c['detection_rate_pct']:.1f}%)")
            row.append(f"{c['avg_ms']:.1f}")
        det_rows.append(row)

    tot_det_row = ["**TOTAL**", f"**{det['total_images']}**", f"**{det['total_gt_quads']}**"]
    for eng in engines:
        te = det["engines"][eng]
        tot_det_row.append(f"**{te['decoded_qr']} ({te['decode_rate_pct']:.1f}%)**")
        tot_det_row.append(f"**{te['detected_quads']} ({te['detection_rate_pct']:.1f}%)**")
        tot_det_row.append(f"**{te['avg_ms']:.2f}**")
    det_rows.append(tot_det_row)
    lines.append(_md_table(det_headers, det_rows, set(range(1, len(det_headers)))))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    py_readers = build_python_readers()
    missing = [e for e in ("zxing-cpp", "wechat-qr", "opencv-qr") if e not in py_readers]
    if missing:
        raise RuntimeError(f"Failed to initialize Python readers: {missing}")

    synthetic_res = evaluate_synthetic(py_readers)
    boofcv_dec_res = evaluate_boofcv_decoding(py_readers)
    boofcv_det_res = evaluate_boofcv_detection(py_readers)

    payload = {
        "meta": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cv2_version": cv2.__version__,
            "engines": ENGINE_ORDER,
            "notes": {
                "timing": "Pure detect+decode time on in-memory image (excludes disk I/O and PNG/JPEG decoding).",
                "qore_flags": "qore --batch-stdin --raw --json --multi (plus --unreadable for BoofCV detection).",
                "boofcv_matching": "1-to-1 greedy matching (IoU >= 0.3 OR mutual centers inside each other's quad).",
            },
        },
        "synthetic": synthetic_res,
        "boofcv_decoding": boofcv_dec_res,
        "boofcv_detection": boofcv_det_res,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "comparison_data.json")
    md_path = os.path.join(RESULTS_DIR, "comparison_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_text = build_markdown_summary(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
