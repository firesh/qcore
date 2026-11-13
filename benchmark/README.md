# Benchmark reproduction

Run these commands from the repository root. The Go library requires no Python packages. The comparison tools use Python 3.10+ and the packages in `benchmark/synthetic/requirements.txt`. The optional deqr reader is used only in the standalone synthetic baseline.

The synthetic benchmark is based on [python-qrcode-synthetic-benchmark](https://github.com/kazuhito00/python-qrcode-synthetic-benchmark). Generate its images and download its WeChat model files with the scripts below. For the BoofCV suites, place the [BoofCV QR v3 dataset](https://boofcv.org/index.php?title=Performance:QrCode) under `benchmark/boofcv/boofcv-v3/` with its `decoding/` and `detection/` directories intact.

```sh
python -m venv .venv
.venv/bin/python -m pip install qrcode Pillow numpy opencv-contrib-python zxing-cpp tabulate
.venv/bin/python benchmark/synthetic/01_download_wechat_models.py
.venv/bin/python benchmark/synthetic/02_generate_test_images.py

mkdir -p bin
go build -o bin/qore ./cmd/qore
go run ./cmd/qore-bench -synthetic benchmark/synthetic/test_images
.venv/bin/python benchmark/run_all_benchmarks.py
.venv/bin/python benchmark/synthetic/03_benchmark.py
.venv/bin/python benchmark/synthetic/04_generate_report.py
```

The four-engine comparison measures detection and decoding on an in-memory image. In the standalone synthetic baseline, third-party timings include image loading and decompression. Run the benchmarks serially when comparing latency.
