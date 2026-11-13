# python-qrcode-synthetic-benchmark
合成テスト画像によるQRコード読み取りライブラリのベンチマーク結果です。<br>
A4相当サイズの画像上にQRコードを配置し、様々な劣化・歪みを加えた780枚の画像で4ライブラリの精度・速度を比較します。

<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/4badfeb1-e59a-4b08-a43c-6662ba8ae9a5" />

# Features
- 65シナリオ × 12データ種類 = 780枚の合成テスト画像を自動生成
- 4つのQRコード読み取りライブラリを統一条件でベンチマーク
- シナリオ別・データ別の詳細分析とHTMLレポート出力

# Benchmark Results

### 全体サマリー
| ライブラリ | 正解数 | 正解率(%) | 合計時間(ms) | 平均時間(ms) |
|---|---|---|---|---|
| **zxing-cpp** | 728 / 780 | 93.3 | 72,405 | 92.8 |
| wechat-qr | 683 / 780 | 87.6 | 43,723 | 56.1 |
| opencv-qr | 404 / 780 | 51.8 | 107,351 | 137.6 |
| deqr | 342 / 780 | 43.8 | 89,788 | 115.1 |

### カテゴリ別傾向
| カテゴリ | 内容 | zxing-cpp | wechat-qr | opencv-qr | deqr |
|---|---|---|---|---|---|
| A. 基本配置 | 書類上の位置・サイズ | ◎ | ○ | △ | △ |
| B. 照明・劣化 | 影・ノイズ・低コントラスト・反転 | ◎ | ◎ | △ | △ |
| C. 歪み・変形 | 台形・樽型・波状・回転 | ○ | ○ | △ | × |
| D. 撮影条件 | スマホ撮影・ブレ・JPEG圧縮・縮小 | ○ | ◎ | × | × |
| E. 複合条件 | 上記の組み合わせ | ○ | ○ | × | × |
| F. 複数QR | 2〜3個・大小混在・歪み付き | ◎ | △ | × | △ |

> 詳細は [results/report.html](https://kazuhito00.github.io/python-qrcode-synthetic-benchmark/results/report.html) を参照してください。

# Test Scenarios
テスト画像は以下の6カテゴリ・65シナリオで構成されています。

| カテゴリ | シナリオ数 | 内容例 |
|---|---|---|
| A. 基本配置 | 6 | 書類中央、右下小QR、表付き書類、机上配置、小QR |
| B. 照明・劣化 | 13 | 影（4方向）、スポットライト、紙テクスチャ、薄印刷、反転、ノイズ |
| C. 歪み・変形 | 21 | 台形歪み（弱〜極強）、見下ろし、横視点、樽型・糸巻き型、波状、円筒面、回転 |
| D. 撮影条件 | 10 | スマホ撮影（良〜悪）、手ブレ、JPEG圧縮、ガウスぼけ、解像度低下 |
| E. 複合条件 | 10 | 机＋歪み、影＋歪み、総合難関 |
| F. 複数QR | 5 | 2個、3個、大小混在、歪み付き複数 |

QRデータは軽量（2〜92文字）8種 + 大容量（265〜1113文字）4種の計12種類です。

# Target Libraries
| ライブラリ | パッケージ | 特徴 |
|---|---|---|
| zxing-cpp | `zxing-cpp` | Googleが開発したZXingのC++移植版。多数のバーコード形式に対応し、高精度なデコード性能を持つ |
| OpenCV QRCodeDetector | `opencv-contrib-python` | OpenCV組込みのQR検出器。追加モデル不要で手軽に利用可能だが、歪みや劣化への耐性は限定的 |
| WeChatQRCode | `opencv-contrib-python` | WeChat由来のCNNベース検出器。事前学習済みCaffeモデルにより劣化画像や撮影条件の変動に強い |
| deqr | `deqr` | quirc / qrdecをバックエンドとして選択可能な軽量ライブラリ。シンプルな条件では高速だが歪みへの耐性は低い |

# Requirements
```
Python 3.10 or later

qrcode[pil]
Pillow
numpy
opencv-contrib-python
zxing-cpp
deqr
tabulate
```

# Installation
```bash
# リポジトリクローン
git clone https://github.com/yourname/python-qrcode-synthetic-benchmark
cd python-qrcode-synthetic-benchmark

# パッケージインストール
pip install -r requirements.txt
```

# Usage

### 1. WeChatQRCodeモデルのダウンロード（初回のみ）
```bash
python 01_download_wechat_models.py
```
`models/wechat_qrcode/` にCaffeモデル4ファイルがダウンロードされます。

### 2. テスト画像生成
```bash
python 02_generate_test_images.py
```
`test_images/` に780枚のPNG画像と `manifest.json` が生成されます。

### 3. ベンチマーク実行
```bash
python 03_benchmark.py
```
`results/` に以下が出力されます：
- `detail.csv` — 全画像×全リーダーの詳細結果
- `summary.txt` — サマリーテキスト
- `report_data.json` — レポート生成用JSON

### 4. HTMLレポート生成
```bash
python 04_generate_report.py
```
[results/report.html](https://kazuhito00.github.io/python-qrcode-synthetic-benchmark/results/report.html) にレポートが生成されます。

# Project Structure
```
01_download_wechat_models.py # WeChatQRCode用モデルDL
02_generate_test_images.py   # テスト画像生成（65シナリオ×12データ）
03_benchmark.py              # ベンチマーク実行（4ライブラリ比較）
04_generate_report.py        # HTMLレポート生成
requirements.txt             # Python依存パッケージ
test_images/                 # 生成されたテスト画像 + manifest.json
models/                      # WeChatQRCode Caffeモデル
results/                     # ベンチマーク結果 + HTMLレポート
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
python-qrcode-synthetic-benchmark is under [Apache-2.0 license](LICENSE).<br>
