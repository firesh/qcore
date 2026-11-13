"""Run the standalone synthetic QR reader benchmark."""

import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from tabulate import tabulate

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def imread_safe(path):
    """日本語パスでも読めるimread"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ============================================================
# 各Engineのラッパー
# ============================================================

class ReaderBase:
    name: str = "base"

    def setup(self):
        return True

    def decode(self, image_path: str) -> list[str]:
        raise NotImplementedError


class ZxingCppReader(ReaderBase):
    name = "zxing-cpp"

    def setup(self):
        try:
            import zxingcpp
            self._zxingcpp = zxingcpp
            return True
        except ImportError:
            return False

    def decode(self, image_path):
        img = imread_safe(image_path)
        if img is None:
            return []
        results = self._zxingcpp.read_barcodes(img)
        return [r.text for r in results if r.text]


class OpenCVQRReader(ReaderBase):
    name = "opencv-qr"

    def setup(self):
        self._detector = cv2.QRCodeDetector()
        return True

    def decode(self, image_path):
        img = imread_safe(image_path)
        if img is None:
            return []
        retval, decoded_info, points, straight_qrcode = self._detector.detectAndDecodeMulti(img)
        if not retval:
            return []
        return [d for d in decoded_info if d]


class WeChatQRReader(ReaderBase):
    name = "wechat-qr"

    def setup(self):
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "models", "wechat_qrcode")
            detect_proto = os.path.join(model_dir, "detect.prototxt")
            detect_model = os.path.join(model_dir, "detect.caffemodel")
            sr_proto = os.path.join(model_dir, "sr.prototxt")
            sr_model = os.path.join(model_dir, "sr.caffemodel")

            if all(os.path.exists(f) for f in [detect_proto, detect_model, sr_proto, sr_model]):
                import shutil
                import tempfile
                tmp_model_dir = os.path.join(tempfile.gettempdir(), "wechat_qrcode_models")
                os.makedirs(tmp_model_dir, exist_ok=True)
                files_map = {}
                for src in [detect_proto, detect_model, sr_proto, sr_model]:
                    dst = os.path.join(tmp_model_dir, os.path.basename(src))
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    files_map[os.path.basename(src)] = dst

                self._detector = cv2.wechat_qrcode_WeChatQRCode(
                    files_map["detect.prototxt"],
                    files_map["detect.caffemodel"],
                    files_map["sr.prototxt"],
                    files_map["sr.caffemodel"],
                )
            else:
                print(f"  [wechat-qr] モデルなしで初期化")
                self._detector = cv2.wechat_qrcode_WeChatQRCode()
            return True
        except (AttributeError, Exception) as e:
            print(f"  [wechat-qr] 初期化失敗: {e}")
            return False

    def decode(self, image_path):
        img = imread_safe(image_path)
        if img is None:
            return []
        results, points = self._detector.detectAndDecode(img)
        return [r for r in results if r]


class DeqrReader(ReaderBase):
    name = "deqr"

    def setup(self):
        try:
            import deqr
            self._deqr = deqr
            return True
        except ImportError:
            return False

    def decode(self, image_path):
        img = imread_safe(image_path)
        if img is None:
            return []
        try:
            decoder = self._deqr.QRdecDecoder()
            results = decoder.decode(img)
            texts = []
            for r in results:
                for entry in r.data_entries:
                    if entry.data:
                        texts.append(entry.data if isinstance(entry.data, str)
                                     else entry.data.decode("utf-8", errors="replace"))
            return texts
        except Exception:
            return []


# ============================================================
# ベンチマーク実行
# ============================================================

ALL_READERS = [
    ZxingCppReader(),
    WeChatQRReader(),
    OpenCVQRReader(),
    DeqrReader(),
]


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    manifest_path = os.path.join(TEST_IMAGES_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("Error: manifest.json is missing. Run 02_generate_test_images.py first.")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    print(f"Test images: {len(manifest)}")

    # Reader 初期化
    active_readers = []
    for reader in ALL_READERS:
        print(f"Initializing: {reader.name} ...", end=" ")
        try:
            ok = reader.setup()
            if ok:
                print("OK")
                active_readers.append(reader)
            else:
                print("SKIP")
        except Exception as e:
            print(f"SKIP ({e})")

    if not active_readers:
        print("No readers available.")
        sys.exit(1)

    print(f"\nActive readers: {[r.name for r in active_readers]}")
    print(f"{'='*80}")

    results = {r.name: {} for r in active_readers}
    total_images = len(manifest)

    for reader in active_readers:
        print(f"\n--- {reader.name} ---")
        for idx, entry in enumerate(manifest):
            filename = entry["filename"]
            expected_list = entry["expected"]
            if isinstance(expected_list, str):
                expected_list = [expected_list]
            image_path = os.path.join(TEST_IMAGES_DIR, filename)

            if (idx + 1) % 40 == 0 or idx == 0:
                print(f"  [{idx+1}/{total_images}] Processing...")

            start = time.perf_counter()
            try:
                decoded_list = reader.decode(image_path)
            except Exception:
                decoded_list = []
            elapsed_ms = (time.perf_counter() - start) * 1000

            expected_sorted = sorted(expected_list)
            found_count = sum(1 for exp in expected_sorted if exp in decoded_list)
            correct = found_count == len(expected_sorted) and len(expected_sorted) > 0

            results[reader.name][filename] = {
                "decoded_list": decoded_list,
                "time_ms": elapsed_ms,
                "correct": correct,
                "found_count": found_count,
                "expected_count": len(expected_sorted),
            }

        correct_count = sum(1 for v in results[reader.name].values() if v["correct"])
        total_time = sum(v["time_ms"] for v in results[reader.name].values())
        print(f"  Result: {correct_count}/{total_images} passed ({correct_count/total_images*100:.1f}%), Total time: {total_time:.0f}ms")

    # ============================================================
    # 集計・レポート出力
    # ============================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    for reader in active_readers:
        r = results[reader.name]
        correct = sum(1 for v in r.values() if v["correct"])
        total_ms = sum(v["time_ms"] for v in r.values())
        avg_ms = total_ms / len(r) if r else 0
        summary_rows.append({
            "Engine": reader.name,
            "Passed": correct,
            "Images": total_images,
            "Pass rate (%)": f"{correct/total_images*100:.1f}",
            "Total time(ms)": f"{total_ms:.0f}",
            "Mean time (ms)": f"{avg_ms:.1f}",
        })
    summary_rows.sort(key=lambda x: float(x["Pass rate (%)"]), reverse=True)
    print(f"\n{'='*80}")
    print("■ Overall results")
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))

    scenario_names = sorted(set(e["scenario"] for e in manifest))
    scenario_table = []
    for sname in scenario_names:
        row = {"Scenario": sname}
        for reader in active_readers:
            relevant = [results[reader.name][e["filename"]]
                        for e in manifest if e["scenario"] == sname]
            correct = sum(1 for v in relevant if v["correct"])
            total = len(relevant)
            row[reader.name] = f"{correct}/{total}"
        scenario_table.append(row)
    print(f"\n■ Pass counts by scenario")
    print(tabulate(scenario_table, headers="keys", tablefmt="github"))

    data_names = sorted(set(e["data_name"] for e in manifest))
    data_table = []
    for dname in data_names:
        row = {"Payload": dname}
        for reader in active_readers:
            relevant = [results[reader.name][e["filename"]]
                        for e in manifest if e["data_name"] == dname]
            correct = sum(1 for v in relevant if v["correct"])
            total = len(relevant)
            row[reader.name] = f"{correct}/{total}"
        data_table.append(row)
    print(f"\n■ Pass counts by payload")
    print(tabulate(data_table, headers="keys", tablefmt="github"))

    # 詳細CSV
    detail_path = os.path.join(RESULTS_DIR, "detail.csv")
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "data_name", "scenario", "expected", "expected_count"]
        for reader in active_readers:
            header.extend([f"{reader.name}_decoded", f"{reader.name}_found",
                           f"{reader.name}_correct", f"{reader.name}_ms"])
        writer.writerow(header)
        for entry in manifest:
            expected = entry["expected"]
            if isinstance(expected, str):
                expected = [expected]
            row = [entry["filename"], entry["data_name"], entry["scenario"],
                   " | ".join(expected), len(expected)]
            for reader in active_readers:
                r = results[reader.name][entry["filename"]]
                row.extend([
                    " | ".join(r["decoded_list"]) if r["decoded_list"] else "",
                    f"{r['found_count']}/{r['expected_count']}",
                    "1" if r["correct"] else "0",
                    f"{r['time_ms']:.1f}",
                ])
            writer.writerow(row)
    print(f"\nDetailed results: {detail_path}")

    # Summaryテキスト
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Synthetic QR Recognition Benchmark\n")
        f.write(f"Test images: {total_images}\n")
        f.write(f"Engines: {len(active_readers)}\n\n")
        f.write("■ Overall results\n")
        f.write(tabulate(summary_rows, headers="keys", tablefmt="github"))
        f.write("\n\n■ Pass counts by scenario\n")
        f.write(tabulate(scenario_table, headers="keys", tablefmt="github"))
        f.write("\n\n■ Pass counts by payload\n")
        f.write(tabulate(data_table, headers="keys", tablefmt="github"))
        f.write("\n")
    print(f"Summary: {summary_path}")

    # JSON (HTMLレポート生成用)
    report_data_path = os.path.join(RESULTS_DIR, "report_data.json")
    with open(report_data_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total_images,
            "readers": [r.name for r in active_readers],
            "summary": summary_rows,
            "scenario_table": scenario_table,
            "data_table": data_table,
        }, f, ensure_ascii=False, indent=2)
    print(f"Report data: {report_data_path}")


if __name__ == "__main__":
    run_benchmark()
