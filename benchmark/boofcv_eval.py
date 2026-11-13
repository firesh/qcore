"""
BoofCV QR v3 benchmark evaluation script.

Evaluates zxing-cpp, wechat-qr, and opencv-qr on:
  1. boofcv-v3/decoding/ (exact & normalized string match against <name>.txt)
  2. boofcv-v3/detection/ across all 16 categories:
     - decoded_qr: QR codes decoded per image (capped at GT quad count)
     - detected_quads: 1-to-1 greedy matching of predicted quads vs GT quads
       (IoU >= 0.3 OR predicted center in GT quad AND GT center in predicted quad)
     - total_ms / avg_ms per image (detector/decoder runtime on in-memory BGR image)

Outputs:
  - benchmark/results/boofcv_baseline.json
  - benchmark/results/boofcv_baseline.txt
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOFCV_DIR = os.path.join(BASE_DIR, "boofcv", "boofcv-v3")
WECHAT_MODEL_DIR = os.path.join(BASE_DIR, "synthetic", "models", "wechat_qrcode")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class Prediction:
    texts: list[str]
    quads: list[np.ndarray]  # each (4, 2) float32


class ReaderBase:
    name: str = "base"

    def setup(self) -> bool:
        raise NotImplementedError

    def predict(self, img: np.ndarray) -> Prediction:
        raise NotImplementedError


class ZxingCppReader(ReaderBase):
    name = "zxing-cpp"

    def setup(self) -> bool:
        import zxingcpp

        self._zxingcpp = zxingcpp
        return True

    def predict(self, img: np.ndarray) -> Prediction:
        results = self._zxingcpp.read_barcodes(img)
        texts: list[str] = []
        quads: list[np.ndarray] = []
        for r in results:
            if r.text:
                texts.append(r.text)
            pos = r.position
            if pos is not None:
                quad = np.array(
                    [
                        [pos.top_left.x, pos.top_left.y],
                        [pos.top_right.x, pos.top_right.y],
                        [pos.bottom_right.x, pos.bottom_right.y],
                        [pos.bottom_left.x, pos.bottom_left.y],
                    ],
                    dtype=np.float32,
                )
                quads.append(quad)
        return Prediction(texts=texts, quads=quads)


class WeChatQRReader(ReaderBase):
    name = "wechat-qr"

    def setup(self) -> bool:
        files = [
            os.path.join(WECHAT_MODEL_DIR, "detect.prototxt"),
            os.path.join(WECHAT_MODEL_DIR, "detect.caffemodel"),
            os.path.join(WECHAT_MODEL_DIR, "sr.prototxt"),
            os.path.join(WECHAT_MODEL_DIR, "sr.caffemodel"),
        ]
        if not all(os.path.exists(f) for f in files):
            raise FileNotFoundError(f"Missing WeChatQRCode model files in {WECHAT_MODEL_DIR}")
        self._detector = cv2.wechat_qrcode_WeChatQRCode(*files)
        return True

    def predict(self, img: np.ndarray) -> Prediction:
        res, points = self._detector.detectAndDecode(img)
        texts = [t for t in res if t]
        quads: list[np.ndarray] = []
        if points is not None:
            for pts in points:
                arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                if len(arr) >= 4:
                    quads.append(arr[:4])
        return Prediction(texts=texts, quads=quads)


class OpenCVQRReader(ReaderBase):
    name = "opencv-qr"

    def setup(self) -> bool:
        self._detector = cv2.QRCodeDetector()
        return True

    def predict(self, img: np.ndarray) -> Prediction:
        retval, decoded_info, points, _ = self._detector.detectAndDecodeMulti(img)
        texts = [d for d in decoded_info if d] if (retval and decoded_info is not None) else []
        quads: list[np.ndarray] = []
        if points is not None:
            pts_arr = np.asarray(points, dtype=np.float32)
            if pts_arr.ndim == 3:
                for pts in pts_arr:
                    if len(pts) >= 4:
                        quads.append(pts[:4])
            elif pts_arr.ndim == 2 and len(pts_arr) >= 4:
                quads.append(pts_arr[:4])
        return Prediction(texts=texts, quads=quads)


def parse_gt_quads(txt_path: str) -> list[np.ndarray]:
    """Parse BoofCV ground-truth .txt files in either Format A (SETS) or Format B (4 lines/quad)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not lines:
        return []

    quads: list[np.ndarray] = []
    if lines[0] == "SETS":
        for line in lines[1:]:
            vals = [float(x) for x in line.split()]
            if len(vals) != 8:
                raise ValueError(f"Expected 8 floats in SETS line in {txt_path}: {line}")
            pts = np.array(vals, dtype=np.float32).reshape(4, 2)
            quads.append(pts)
    else:
        if len(lines) % 4 != 0:
            raise ValueError(f"Expected multiple of 4 lines in {txt_path}, got {len(lines)}")
        pts_all = [[float(x) for x in line.split()] for line in lines]
        for i in range(0, len(pts_all), 4):
            quads.append(np.array(pts_all[i : i + 4], dtype=np.float32))
    return quads


def _to_convex_poly(quad: np.ndarray) -> np.ndarray:
    """Return convex hull vertices in consistent order for IoU / point-in-polygon."""
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return cv2.convexHull(pts).reshape(-1, 2)


def _quad_center(quad: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _point_in_poly(pt: tuple[float, float], poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly, pt, False) >= 0.0


def quad_iou(q1: np.ndarray, q2: np.ndarray) -> tuple[float, bool]:
    """Return (IoU, mutual_center_inside) between two quads."""
    p1 = _to_convex_poly(q1)
    p2 = _to_convex_poly(q2)

    a1 = float(cv2.contourArea(p1))
    a2 = float(cv2.contourArea(p2))
    if a1 <= 0.0 or a2 <= 0.0:
        iou = 0.0
    else:
        inter_area, _ = cv2.intersectConvexConvex(p1, p2)
        union = a1 + a2 - float(inter_area)
        iou = float(inter_area) / union if union > 0.0 else 0.0

    c1 = _quad_center(q1)
    c2 = _quad_center(q2)
    mutual_center = _point_in_poly(c1, p2) and _point_in_poly(c2, p1)
    return iou, mutual_center


def match_quads_greedy(pred_quads: list[np.ndarray], gt_quads: list[np.ndarray]) -> int:
    """1-to-1 greedy matching where IoU >= 0.3 OR mutual centers lie inside each other's quad."""
    candidates: list[tuple[float, int, int]] = []
    for pi, pq in enumerate(pred_quads):
        for gi, gq in enumerate(gt_quads):
            iou, mutual = quad_iou(pq, gq)
            if iou >= 0.3 or mutual:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched = 0
    for _, pi, gi in candidates:
        if pi not in used_pred and gi not in used_gt:
            used_pred.add(pi)
            used_gt.add(gi)
            matched += 1
    return matched


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def list_images(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(IMAGE_EXTS)
    )


def evaluate_decoding(readers: Iterable[ReaderBase]) -> dict:
    decoding_dir = os.path.join(BOOFCV_DIR, "decoding")
    image_paths = list_images(decoding_dir)
    dataset = []
    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            gt_raw = f.read()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        dataset.append((os.path.basename(img_path), img, gt_raw, normalize_text(gt_raw)))

    results: dict[str, dict] = {}
    for reader in readers:
        correct_norm = 0
        correct_exact = 0
        total_ms = 0.0
        per_image = []

        for fname, img, gt_raw, gt_norm in dataset:
            t0 = time.perf_counter()
            try:
                pred = reader.predict(img)
            except Exception:
                pred = Prediction(texts=[], quads=[])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_ms += elapsed_ms

            matched_exact = any(t == gt_raw for t in pred.texts)
            matched_norm = any(normalize_text(t) == gt_norm for t in pred.texts)
            if matched_exact:
                correct_exact += 1
            if matched_norm:
                correct_norm += 1

            per_image.append(
                {
                    "filename": fname,
                    "matched_norm": matched_norm,
                    "matched_exact": matched_exact,
                    "decoded_count": len(pred.texts),
                    "time_ms": round(elapsed_ms, 3),
                }
            )

        n = len(dataset)
        results[reader.name] = {
            "total_images": n,
            "correct_norm": correct_norm,
            "accuracy_norm_pct": round(correct_norm / n * 100.0, 2) if n else 0.0,
            "correct_exact": correct_exact,
            "accuracy_exact_pct": round(correct_exact / n * 100.0, 2) if n else 0.0,
            "total_ms": round(total_ms, 2),
            "avg_ms": round(total_ms / n, 2) if n else 0.0,
            "per_image": per_image,
        }
    return results


def evaluate_detection(readers: Iterable[ReaderBase]) -> dict:
    detection_root = os.path.join(BOOFCV_DIR, "detection")
    categories = sorted(
        d
        for d in os.listdir(detection_root)
        if os.path.isdir(os.path.join(detection_root, d))
    )

    cat_datasets: dict[str, list[tuple[str, np.ndarray, list[np.ndarray]]]] = {}
    total_images = 0
    total_gt_quads = 0

    for cat in categories:
        cat_dir = os.path.join(detection_root, cat)
        items = []
        for img_path in list_images(cat_dir):
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            gt_quads = parse_gt_quads(txt_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            items.append((os.path.basename(img_path), img, gt_quads))
            total_images += 1
            total_gt_quads += len(gt_quads)
        cat_datasets[cat] = items

    reader_results: dict[str, dict] = {}
    for reader in readers:
        print(f"[detection] Evaluating {reader.name} on {total_images} images ({total_gt_quads} GT quads)...")
        by_category: dict[str, dict] = {}
        overall_decoded = 0
        overall_detected = 0
        overall_ms = 0.0

        for cat in categories:
            items = cat_datasets[cat]
            cat_gt = sum(len(gt) for _, _, gt in items)
            cat_decoded = 0
            cat_detected = 0
            cat_ms = 0.0

            for _, img, gt_quads in items:
                t0 = time.perf_counter()
                try:
                    pred = reader.predict(img)
                except Exception:
                    pred = Prediction(texts=[], quads=[])
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                cat_ms += elapsed_ms

                gt_count = len(gt_quads)
                cat_decoded += min(len(pred.texts), gt_count)
                cat_detected += match_quads_greedy(pred.quads, gt_quads)

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

        reader_results[reader.name] = {
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

    return {
        "categories": categories,
        "total_images": total_images,
        "total_gt_quads": total_gt_quads,
        "readers": reader_results,
    }


def format_report(decoding_res: dict, detection_res: dict) -> str:
    lines: list[str] = []
    lines.append("BoofCV QR v3 Baseline Evaluation")
    lines.append("=" * 80)
    lines.append("Timing note: measures detector/decoder call on preloaded BGR image (excludes disk I/O).")
    lines.append("")

    # Decoding summary
    lines.append("1. Decoding Benchmark (boofcv-v3/decoding)")
    dec_rows = []
    for name, r in decoding_res.items():
        dec_rows.append(
            {
                "Library": name,
                "Correct (norm)": f"{r['correct_norm']}/{r['total_images']}",
                "Accuracy % (norm)": f"{r['accuracy_norm_pct']:.1f}",
                "Correct (raw)": f"{r['correct_exact']}/{r['total_images']}",
                "Total ms": f"{r['total_ms']:.1f}",
                "Avg ms": f"{r['avg_ms']:.2f}",
            }
        )
    lines.append(tabulate(dec_rows, headers="keys", tablefmt="github"))
    lines.append("")

    # Detection overall summary
    lines.append("2. Detection + Decoding Benchmark Overall (boofcv-v3/detection, 16 categories)")
    det_rows = []
    for name, r in detection_res["readers"].items():
        det_rows.append(
            {
                "Library": name,
                "Images": r["total_images"],
                "GT Quads": r["total_gt_quads"],
                "Decoded QR": f"{r['decoded_qr']}/{r['total_gt_quads']}",
                "Decode %": f"{r['decode_rate_pct']:.1f}",
                "Detected Quads": f"{r['detected_quads']}/{r['total_gt_quads']}",
                "Detection %": f"{r['detection_rate_pct']:.1f}",
                "Total ms": f"{r['total_ms']:.0f}",
                "Avg ms": f"{r['avg_ms']:.1f}",
            }
        )
    lines.append(tabulate(det_rows, headers="keys", tablefmt="github"))
    lines.append("")

    # Per-category table
    lines.append("3. Detection + Decoding by Category (decoded / detected / GT)")
    reader_names = list(detection_res["readers"].keys())
    cat_rows = []
    for cat in detection_res["categories"]:
        first_cat = detection_res["readers"][reader_names[0]]["categories"][cat]
        row = {
            "Category": cat,
            "Imgs": first_cat["images"],
            "GT": first_cat["gt_quads"],
        }
        for name in reader_names:
            c = detection_res["readers"][name]["categories"][cat]
            row[f"{name} dec"] = f"{c['decoded_qr']} ({c['decode_rate_pct']:.1f}%)"
            row[f"{name} det"] = f"{c['detected_quads']} ({c['detection_rate_pct']:.1f}%)"
        cat_rows.append(row)
    lines.append(tabulate(cat_rows, headers="keys", tablefmt="github"))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    readers: list[ReaderBase] = []
    for cls in (ZxingCppReader, WeChatQRReader, OpenCVQRReader):
        reader = cls()
        if reader.setup():
            readers.append(reader)

    decoding_res = evaluate_decoding(readers)
    detection_res = evaluate_detection(readers)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "boofcv_baseline.json")
    txt_path = os.path.join(RESULTS_DIR, "boofcv_baseline.txt")

    payload = {
        "dataset": "boofcv-v3",
        "decoding": decoding_res,
        "detection": detection_res,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    report_txt = format_report(decoding_res, detection_res)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print("\n" + report_txt)
    print(f"Saved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
