"""
テスト用QRコード画像を生成するスクリプト（実環境想定・拡張版）

カテゴリ:
  A. 基本配置 — 書類上のさまざまな位置にQR
  B. 照明・劣化 — 影、照明ムラ、シミ、薄印刷、反転
  C. 歪み・変形 — 台形歪み（弱〜極強）、樽型・糸巻き型歪み、波状歪み、回転、傾き
  D. 撮影条件 — スマホ撮影、手ブレ、ピンぼけ、JPEG圧縮、解像度低下
  E. 複合条件 — 上記の組み合わせ
  F. 複数QR — 2個、3個、大小混在、歪み付き
"""

import json
import math
import os
import random

import cv2
import numpy as np
import qrcode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")

A4_W, A4_H = 1240, 1754

# ============================================================
# QRコード生成データ
# ============================================================
QR_DATA = [
    # --- 軽量 (2〜91文字) ---
    ("url",       "https://example.com/product/12345"),
    ("long_url",  "https://example.com/very/long/path?param1=value1&param2=value2&param3=value3&token=abc123xyz"),
    ("japanese",  "日本語テスト：QRコード読み取りベンチマーク2025"),
    ("numeric",   "0123456789012345678901234567890123456789"),
    ("short",     "OK"),
    ("vcard",     "BEGIN:VCARD\nVERSION:3.0\nFN:Taro Yamada\nTEL:090-1234-5678\nEND:VCARD"),
    ("wifi",      "WIFI:T:WPA;S:MyNetwork;P:password123;;"),
    ("uuid",      "550e8400-e29b-41d4-a716-446655440000"),
    # --- 大容量 (情報密度テスト) ---
    ("dense_300", '{"id":12345,"name":"Benchmark Test Product","description":"This is a fairly long JSON payload to test QR code reading with moderate data density","price":9800,"currency":"JPY","tags":["electronics","gadget","sale"],"url":"https://example.com/products/12345?ref=qr"}'),
    ("dense_600", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Vivamus lacinia odio vitae vestibulum vestibulum. Praesent commodo cursus magna, vel scelerisque nisl consectetur et volutpat."),
    ("dense_1200", "BEGIN:VEVENT\nDTSTART:20250401T090000Z\nDTEND:20250401T170000Z\nSUMMARY:Annual Technology Conference 2025 - Building the Future with AI and Machine Learning\nDESCRIPTION:Join us for a full day of presentations and workshops covering the latest advances in artificial intelligence, machine learning, computer vision, and natural language processing. Featured speakers include leading researchers from top universities and industry practitioners who will share their insights on practical applications of AI in healthcare, finance, manufacturing, and education. Morning sessions focus on theoretical foundations while afternoon workshops provide hands-on experience with popular frameworks and tools. Lunch and refreshments will be provided. Please register in advance as space is limited to 200 attendees. Early bird discount available until March 15.\nLOCATION:Tokyo International Forum, Hall B7, 3-5-1 Marunouchi, Chiyoda-ku, Tokyo 100-0005\nORGANIZER:mailto:events@techconf2025.example.com\nURL:https://techconf2025.example.com/register?event=annual&year=2025&track=ai-ml&session=full-day&code=EARLYBIRD2025\nEND:VEVENT"),
    ("dense_jp_500", "令和7年度第3回定例会議の議事録。出席者：山田太郎（議長）、佐藤花子、鈴木一郎、田中美咲、高橋健太、伊藤裕子、渡辺大輔、中村美穂。議題1：来年度の予算計画について。各部署からの予算要求を取りまとめた結果、総額は前年度比12%増の3億4500万円となった。主な増加要因はDX推進関連費用とセキュリティ強化費用である。議題2：新規プロジェクトの進捗報告。AIを活用した品質管理システムの開発は計画通り進行中であり、来月末にプロトタイプが完成予定。議題3：社内研修制度の改定について。次回会議は4月15日14時からを予定。以上。"),
]


# ============================================================
# ユーティリティ
# ============================================================

def imwrite_safe(path, img):
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(path)
    return result


def generate_qr_image(data: str, box_size: int = 10, border: int = 4) -> np.ndarray:
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
                        box_size=box_size, border=border)
    qr.add_data(data)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def create_blank_a4(bg_color=(255, 255, 255)):
    return np.full((A4_H, A4_W, 3), bg_color, dtype=np.uint8)


def draw_dummy_text_lines(img, y_start=80, y_end=None, line_height=30, margin_x=80):
    if y_end is None:
        y_end = img.shape[0] - 80
    y = y_start
    while y < y_end:
        line_len = random.randint(int((img.shape[1] - 2 * margin_x) * 0.4),
                                  img.shape[1] - 2 * margin_x)
        color = random.randint(40, 80)
        thickness = random.choice([1, 1, 1, 2])
        cv2.line(img, (margin_x, y), (margin_x + line_len, y),
                 (color, color, color), thickness)
        y += line_height + random.randint(-3, 5)
    return img


def draw_heading(img, y=40, margin_x=80):
    cv2.line(img, (margin_x, y), (margin_x + random.randint(200, 400), y), (20, 20, 20), 3)
    return img


def draw_table_grid(img, x, y, rows, cols, cell_w, cell_h):
    for r in range(rows + 1):
        cv2.line(img, (x, y + r * cell_h), (x + cols * cell_w, y + r * cell_h), (100, 100, 100), 1)
    for c in range(cols + 1):
        cv2.line(img, (x + c * cell_w, y), (x + c * cell_w, y + rows * cell_h), (100, 100, 100), 1)
    return img


def place_qr(canvas, qr_img, x, y):
    qh, qw = qr_img.shape[:2]
    ch, cw = canvas.shape[:2]
    x2, y2 = min(x + qw, cw), min(y + qh, ch)
    x, y = max(x, 0), max(y, 0)
    canvas[y:y2, x:x2] = qr_img[:y2 - y, :x2 - x]
    return canvas


def make_doc_with_qr(qr_data, box_size=7, qr_x=None, qr_y=550, bg=(255, 255, 255)):
    """標準的な書類+QR画像を生成するヘルパー"""
    canvas = create_blank_a4(bg)
    draw_heading(canvas)
    draw_dummy_text_lines(canvas, y_start=90, y_end=min(qr_y - 40, 500))
    qr_img = generate_qr_image(qr_data, box_size=box_size)
    qh, qw = qr_img.shape[:2]
    if qr_x is None:
        qr_x = (A4_W - qw) // 2
    place_qr(canvas, qr_img, qr_x, qr_y)
    draw_dummy_text_lines(canvas, y_start=qr_y + qh + 30, y_end=A4_H - 80)
    return canvas


# ============================================================
# 後処理エフェクト
# ============================================================

def add_shadow_gradient(img, direction="left"):
    h, w = img.shape[:2]
    if direction == "left":
        grad = np.linspace(0.4, 1.0, w).reshape(1, w, 1)
    elif direction == "right":
        grad = np.linspace(1.0, 0.4, w).reshape(1, w, 1)
    elif direction == "top":
        grad = np.linspace(0.4, 1.0, h).reshape(h, 1, 1)
    elif direction == "bottom":
        grad = np.linspace(1.0, 0.4, h).reshape(h, 1, 1)
    else:  # diagonal
        gx = np.linspace(0.5, 1.0, w)
        gy = np.linspace(0.5, 1.0, h)
        grad = np.outer(gy, gx)[:, :, np.newaxis]
    return np.clip(img.astype(np.float64) * grad, 0, 255).astype(np.uint8)


def add_spotlight(img, cx=None, cy=None, radius=None):
    h, w = img.shape[:2]
    cx = cx or random.randint(w // 4, 3 * w // 4)
    cy = cy or random.randint(h // 4, 3 * h // 4)
    radius = radius or max(h, w) * 0.6
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip(1.0 - dist / radius * 0.5, 0.3, 1.0)[:, :, np.newaxis]
    return np.clip(img.astype(np.float64) * mask, 0, 255).astype(np.uint8)


def add_paper_texture(img, intensity=15):
    noise = np.random.normal(0, intensity, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_wrinkle_lines(img, n_lines=5):
    h, w = img.shape[:2]
    result = img.copy()
    for _ in range(n_lines):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2 = x1 + random.randint(-w // 3, w // 3)
        y2 = y1 + random.randint(-h // 3, h // 3)
        color = random.randint(180, 220)
        cv2.line(result, (x1, y1), (x2, y2), (color, color, color), 1, cv2.LINE_AA)
    return result


def perspective_warp(img, strength=0.1):
    h, w = img.shape[:2]
    s = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * s * random.uniform(0.5, 1.5), h * s * random.uniform(0.5, 1.5)],
        [w * (1 - s * random.uniform(0.5, 1.5)), h * s * random.uniform(0, 0.5)],
        [w * (1 - s * random.uniform(0, 0.5)), h * (1 - s * random.uniform(0, 0.5))],
        [w * s * random.uniform(0, 0.5), h * (1 - s * random.uniform(0.5, 1.5))],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(200, 200, 200))


def perspective_topdown(img, pitch_deg=30):
    """上から見下ろす角度のシミュレーション"""
    h, w = img.shape[:2]
    f = w  # 焦点距離相当
    pitch = math.radians(pitch_deg)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    squeeze = math.cos(pitch)
    offset_top = h * (1 - squeeze) * 0.5
    inset = w * math.sin(pitch) * 0.15
    dst = np.float32([
        [inset, offset_top],
        [w - inset, offset_top],
        [w, h],
        [0, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(200, 200, 200))


def perspective_side(img, yaw_deg=25):
    """横から見た角度のシミュレーション"""
    h, w = img.shape[:2]
    squeeze = math.cos(math.radians(yaw_deg))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inset = h * (1 - squeeze) * 0.3
    dst = np.float32([
        [0, inset],
        [w, 0],
        [w, h],
        [0, h - inset],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(200, 200, 200))


def barrel_distortion(img, k1=-0.3):
    h, w = img.shape[:2]
    K = np.eye(3, dtype=np.float64)
    K[0, 0], K[1, 1] = w, h
    K[0, 2], K[1, 2] = w / 2, h / 2
    dist_coeffs = np.array([k1, 0.05, 0, 0, 0], dtype=np.float64)
    return cv2.undistort(img, K, dist_coeffs)


def pincushion_distortion(img):
    return barrel_distortion(img, k1=0.3)


def wave_distortion(img, amplitude=10, frequency=0.02):
    """波状の歪み"""
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x + amplitude * math.sin(frequency * y * 2 * math.pi)
            map_y[y, x] = y + amplitude * math.sin(frequency * x * 2 * math.pi)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def cylindrical_warp(img, strength=300):
    """円筒面に貼りつけたような歪み（丸まった紙のシミュレーション）"""
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    cx = w / 2
    for y in range(h):
        for x in range(w):
            dx = x - cx
            theta = dx / strength
            map_x[y, x] = cx + strength * math.sin(theta)
            scale = math.cos(theta)
            map_y[y, x] = h / 2 + (y - h / 2) * scale
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(230, 230, 230))


def add_desk_background(img, desk_color=None):
    if desk_color is None:
        desk_color = random.choice([
            (180, 200, 220), (100, 120, 140), (200, 200, 200), (160, 180, 170),
        ])
    h, w = img.shape[:2]
    margin = 60
    canvas = np.full((h + margin * 2, w + margin * 2, 3), desk_color, dtype=np.uint8)
    desk_noise = np.random.normal(0, 8, canvas.shape)
    canvas = np.clip(canvas.astype(np.float64) + desk_noise, 0, 255).astype(np.uint8)
    canvas[margin:margin + h, margin:margin + w] = img
    cv2.line(canvas, (margin + w, margin + 3), (margin + w, margin + h + 3), (80, 80, 80), 3)
    cv2.line(canvas, (margin + 3, margin + h), (margin + w + 3, margin + h), (80, 80, 80), 3)
    return canvas


def simulate_phone_capture(img, blur_sigma=1.0, jpeg_quality=60, downscale=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * downscale), int(h * downscale)), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1
        small = cv2.GaussianBlur(small, (ksize, ksize), blur_sigma)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_motion_blur(img, size=15, angle=0):
    kernel = np.zeros((size, size))
    center = size // 2
    cos_val, sin_val = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for i in range(size):
        offset = i - center
        x, y = int(center + offset * cos_val), int(center + offset * sin_val)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def rotate_image(img, angle, border_color=(200, 200, 200)):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=border_color)


def jpeg_compress(img, quality):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_stains(img, qr_x, qr_y, qr_w, qr_h, n=3):
    """QR周辺にシミを追加"""
    result = img.copy()
    for _ in range(n):
        cx = qr_x + random.randint(-50, qr_w + 50)
        cy = qr_y + random.randint(-50, qr_h + 50)
        r = random.randint(20, 60)
        color = random.choice([(180, 200, 220), (200, 210, 230), (190, 195, 200)])
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1)
        alpha = random.uniform(0.2, 0.5)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    return result


# ============================================================
# シナリオ定義（全て list of (scenario_name, img, expected_list) を返す）
# ============================================================

# --- A. 基本配置 ---

def s_doc_center(d, n):
    return [("A01_doc_center", make_doc_with_qr(d), [d])]

def s_doc_corner_small(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, A4_H - 200)
    qr = generate_qr_image(d, box_size=3, border=2)
    place_qr(c, qr, A4_W - qr.shape[1] - 60, A4_H - qr.shape[0] - 60)
    return [("A02_doc_corner_small", c, [d])]

def s_doc_top_right(d, n):
    c = create_blank_a4()
    qr = generate_qr_image(d, box_size=5)
    place_qr(c, qr, A4_W - qr.shape[1] - 60, 50)
    draw_dummy_text_lines(c, 50 + qr.shape[0] + 30, A4_H - 80)
    return [("A03_doc_top_right", c, [d])]

def s_doc_with_table(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, 300)
    draw_table_grid(c, 80, 320, 8, 5, 150, 35)
    qr = generate_qr_image(d, box_size=5)
    place_qr(c, qr, A4_W - qr.shape[1] - 60, 350)
    draw_dummy_text_lines(c, 650, A4_H - 80)
    return [("A04_doc_with_table", c, [d])]

def s_on_desk(d, n):
    c = make_doc_with_qr(d, box_size=6, qr_x=80, qr_y=540)
    return [("A05_on_desk", add_desk_background(c), [d])]

def s_tiny_qr(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, A4_H - 120)
    qr = generate_qr_image(d, box_size=2, border=2)
    place_qr(c, qr, A4_W - qr.shape[1] - 40, A4_H - qr.shape[0] - 40)
    return [("A06_tiny_qr", c, [d])]

# --- B. 照明・劣化 ---

def s_shadow_left(d, n):
    return [("B01_shadow_left", add_shadow_gradient(make_doc_with_qr(d), "left"), [d])]

def s_shadow_right(d, n):
    return [("B02_shadow_right", add_shadow_gradient(make_doc_with_qr(d), "right"), [d])]

def s_shadow_diagonal(d, n):
    return [("B03_shadow_diagonal", add_shadow_gradient(make_doc_with_qr(d), "diagonal"), [d])]

def s_shadow_top(d, n):
    return [("B04_shadow_top", add_shadow_gradient(make_doc_with_qr(d), "top"), [d])]

def s_spotlight(d, n):
    return [("B05_spotlight", add_spotlight(make_doc_with_qr(d)), [d])]

def s_spotlight_off_center(d, n):
    img = make_doc_with_qr(d)
    return [("B06_spotlight_offset", add_spotlight(img, cx=200, cy=300, radius=500), [d])]

def s_paper_texture(d, n):
    c = make_doc_with_qr(d, bg=(248, 245, 240))
    return [("B07_paper_texture", add_wrinkle_lines(add_paper_texture(c, 20), 8), [d])]

def s_stained(d, n):
    c = make_doc_with_qr(d)
    qr = generate_qr_image(d, box_size=7)
    qh, qw = qr.shape[:2]
    qx, qy = (A4_W - qw) // 2, 550
    return [("B08_stained", add_stains(c, qx, qy, qw, qh, 4), [d])]

def s_low_contrast_print(d, n):
    c = create_blank_a4((245, 245, 245))
    draw_dummy_text_lines(c, 60, 500)
    qr = generate_qr_image(d, box_size=7)
    qr = np.clip(qr.astype(np.int16) + 120, 0, 255).astype(np.uint8)
    place_qr(c, qr, (A4_W - qr.shape[1]) // 2, 550)
    draw_dummy_text_lines(c, 550 + qr.shape[0] + 30, A4_H - 80)
    return [("B09_low_contrast", c, [d])]

def s_very_low_contrast(d, n):
    c = create_blank_a4((240, 240, 240))
    qr = generate_qr_image(d, box_size=7)
    qr = np.clip(qr.astype(np.int16) + 160, 0, 255).astype(np.uint8)
    place_qr(c, qr, (A4_W - qr.shape[1]) // 2, (A4_H - qr.shape[0]) // 2)
    return [("B10_very_low_contrast", c, [d])]

def s_inverted(d, n):
    c = create_blank_a4((30, 30, 30))
    qr = cv2.bitwise_not(generate_qr_image(d, box_size=7))
    place_qr(c, qr, (A4_W - qr.shape[1]) // 2, (A4_H - qr.shape[0]) // 2)
    return [("B11_inverted", c, [d])]

def s_colored_paper(d, n):
    bg = random.choice([(255, 255, 200), (200, 255, 200), (200, 230, 255), (255, 220, 220)])
    c = make_doc_with_qr(d, bg=bg)
    return [("B12_colored_paper", c, [d])]

def s_heavy_noise(d, n):
    return [("B13_heavy_noise", add_gaussian_noise(make_doc_with_qr(d), 50), [d])]

# --- C. 歪み・変形 ---

def s_perspective_mild(d, n):
    return [("C01_persp_mild", perspective_warp(make_doc_with_qr(d), 0.06), [d])]

def s_perspective_moderate(d, n):
    return [("C02_persp_moderate", perspective_warp(make_doc_with_qr(d), 0.12), [d])]

def s_perspective_strong(d, n):
    return [("C03_persp_strong", perspective_warp(make_doc_with_qr(d), 0.18), [d])]

def s_perspective_extreme(d, n):
    return [("C04_persp_extreme", perspective_warp(make_doc_with_qr(d), 0.25), [d])]

def s_topdown_20(d, n):
    return [("C05_topdown_20deg", perspective_topdown(make_doc_with_qr(d), 20), [d])]

def s_topdown_40(d, n):
    return [("C06_topdown_40deg", perspective_topdown(make_doc_with_qr(d), 40), [d])]

def s_topdown_55(d, n):
    return [("C07_topdown_55deg", perspective_topdown(make_doc_with_qr(d), 55), [d])]

def s_side_20(d, n):
    return [("C08_side_20deg", perspective_side(make_doc_with_qr(d), 20), [d])]

def s_side_40(d, n):
    return [("C09_side_40deg", perspective_side(make_doc_with_qr(d), 40), [d])]

def s_barrel(d, n):
    return [("C10_barrel", barrel_distortion(make_doc_with_qr(d), -0.3), [d])]

def s_barrel_strong(d, n):
    return [("C11_barrel_strong", barrel_distortion(make_doc_with_qr(d), -0.6), [d])]

def s_pincushion(d, n):
    return [("C12_pincushion", pincushion_distortion(make_doc_with_qr(d)), [d])]

def s_wave(d, n):
    return [("C13_wave", wave_distortion(make_doc_with_qr(d), amplitude=2, frequency=0.005), [d])]

def s_wave_strong(d, n):
    return [("C14_wave_strong", wave_distortion(make_doc_with_qr(d), amplitude=2.5, frequency=0.005), [d])]

def s_cylindrical(d, n):
    return [("C15_cylindrical", cylindrical_warp(make_doc_with_qr(d), 400), [d])]

def s_cylindrical_tight(d, n):
    return [("C16_cylindrical_tight", cylindrical_warp(make_doc_with_qr(d), 350), [d])]

def s_rotated_5(d, n):
    return [("C17_rotate_5", rotate_image(make_doc_with_qr(d), 5), [d])]

def s_rotated_15(d, n):
    return [("C18_rotate_15", rotate_image(make_doc_with_qr(d), 15), [d])]

def s_rotated_45(d, n):
    return [("C19_rotate_45", rotate_image(make_doc_with_qr(d), 45), [d])]

def s_rotated_90(d, n):
    return [("C20_rotate_90", rotate_image(make_doc_with_qr(d), 90), [d])]

def s_rotated_180(d, n):
    return [("C21_rotate_180", rotate_image(make_doc_with_qr(d), 180), [d])]

# --- D. 撮影条件 ---

def s_phone_good(d, n):
    c = add_desk_background(make_doc_with_qr(d))
    return [("D01_phone_good", simulate_phone_capture(c, 0.5, 75, 0.6), [d])]

def s_phone_medium(d, n):
    c = add_desk_background(make_doc_with_qr(d))
    return [("D02_phone_medium", simulate_phone_capture(c, 1.0, 50, 0.45), [d])]

def s_phone_bad(d, n):
    c = add_shadow_gradient(make_doc_with_qr(d), "diagonal")
    c = add_desk_background(c)
    return [("D03_phone_bad", simulate_phone_capture(c, 1.0, 50, 0.45), [d])]

def s_motion_blur_light(d, n):
    angle = random.randint(-20, 20)
    return [("D04_motion_blur_light", add_motion_blur(make_doc_with_qr(d), 8, angle), [d])]

def s_motion_blur_heavy(d, n):
    angle = random.randint(-20, 20)
    return [("D05_motion_blur_heavy", add_motion_blur(make_doc_with_qr(d), 12, angle), [d])]

def s_jpeg_q15(d, n):
    return [("D06_jpeg_q15", jpeg_compress(make_doc_with_qr(d), 15), [d])]

def s_jpeg_q5(d, n):
    return [("D07_jpeg_q5", jpeg_compress(make_doc_with_qr(d), 5), [d])]

def s_blur_gaussian(d, n):
    img = make_doc_with_qr(d)
    return [("D08_blur_gaussian", cv2.GaussianBlur(img, (9, 9), 0), [d])]

def s_downscale_30pct(d, n):
    img = make_doc_with_qr(d)
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.3), int(h * 0.3)), interpolation=cv2.INTER_AREA)
    return [("D09_downscale_30pct", small, [d])]

def s_downscale_15pct(d, n):
    img = make_doc_with_qr(d)
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.15), int(h * 0.15)), interpolation=cv2.INTER_AREA)
    return [("D10_downscale_15pct", small, [d])]

# --- E. 複合条件 ---

def s_desk_perspective(d, n):
    c = add_desk_background(make_doc_with_qr(d, box_size=6))
    return [("E01_desk_persp", perspective_warp(c, 0.10), [d])]

def s_desk_topdown(d, n):
    c = add_desk_background(make_doc_with_qr(d, box_size=6))
    return [("E02_desk_topdown", perspective_topdown(c, 35), [d])]

def s_shadow_perspective(d, n):
    c = add_shadow_gradient(make_doc_with_qr(d), "left")
    return [("E03_shadow_persp", perspective_warp(c, 0.12), [d])]

def s_texture_barrel(d, n):
    c = make_doc_with_qr(d, bg=(248, 245, 240))
    c = add_paper_texture(c, 18)
    return [("E04_texture_barrel", barrel_distortion(c, -0.3), [d])]

def s_noise_perspective(d, n):
    c = add_gaussian_noise(make_doc_with_qr(d), 35)
    return [("E05_noise_persp", perspective_warp(c, 0.10), [d])]

def s_phone_perspective(d, n):
    c = add_desk_background(make_doc_with_qr(d))
    c = perspective_warp(c, 0.10)
    return [("E06_phone_persp", simulate_phone_capture(c, 0.8, 55, 0.5), [d])]

def s_shadow_wave(d, n):
    c = add_shadow_gradient(make_doc_with_qr(d), "diagonal")
    return [("E07_shadow_wave", wave_distortion(c, 2, 0.005), [d])]

def s_combined_hard(d, n):
    c = make_doc_with_qr(d, box_size=6, bg=(250, 248, 242))
    c = add_paper_texture(c, 15)
    c = add_wrinkle_lines(c, 4)
    c = add_shadow_gradient(c, "diagonal")
    c = add_desk_background(c)
    c = perspective_warp(c, 0.10)
    return [("E08_combined_hard", simulate_phone_capture(c, 1.0, 50, 0.5), [d])]

def s_combined_extreme(d, n):
    c = make_doc_with_qr(d, box_size=6, bg=(248, 245, 240))
    c = add_paper_texture(c, 15)
    c = add_shadow_gradient(c, "left")
    c = barrel_distortion(c, -0.15)
    c = add_desk_background(c)
    c = perspective_warp(c, 0.10)
    return [("E09_combined_extreme", simulate_phone_capture(c, 0.8, 55, 0.5), [d])]

def s_cylindrical_shadow_desk(d, n):
    c = make_doc_with_qr(d, box_size=6)
    c = cylindrical_warp(c, 350)
    c = add_shadow_gradient(c, "right")
    return [("E10_cylinder_shadow", add_desk_background(c), [d])]

# --- F. 複数QR ---

def s_multi_two(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, 400)
    d2 = "https://example.com/second-qr-code"
    q1, q2 = generate_qr_image(d, box_size=5), generate_qr_image(d2, box_size=5)
    place_qr(c, q1, 80, 450)
    place_qr(c, q2, A4_W - q2.shape[1] - 80, 450)
    draw_dummy_text_lines(c, max(450 + q1.shape[0], 450 + q2.shape[0]) + 30, A4_H - 80)
    return [("F01_multi_two", c, sorted([d, d2]))]

def s_multi_three(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, 300)
    d2, d3 = "TEL:03-1234-5678", "mailto:info@example.com"
    q1 = generate_qr_image(d, box_size=4)
    q2, q3 = generate_qr_image(d2, box_size=4), generate_qr_image(d3, box_size=4)
    place_qr(c, q1, (A4_W - q1.shape[1]) // 2, 340)
    y2 = 340 + q1.shape[0] + 80
    place_qr(c, q2, 100, y2)
    place_qr(c, q3, A4_W - q3.shape[1] - 100, y2)
    draw_dummy_text_lines(c, y2 + max(q2.shape[0], q3.shape[0]) + 30, A4_H - 80)
    return [("F02_multi_three", c, sorted([d, d2, d3]))]

def s_multi_sizes(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, 350)
    d2 = "ITEM-12345"
    qb, qs = generate_qr_image(d, box_size=8), generate_qr_image(d2, box_size=3, border=2)
    place_qr(c, qb, 80, 400)
    place_qr(c, qs, A4_W - qs.shape[1] - 50, A4_H - qs.shape[0] - 50)
    draw_dummy_text_lines(c, 400 + qb.shape[0] + 30, A4_H - qs.shape[0] - 80)
    return [("F03_multi_sizes", c, sorted([d, d2]))]

def s_multi_desk_persp(d, n):
    c = create_blank_a4()
    draw_heading(c); draw_dummy_text_lines(c, 90, 400)
    d2 = "SCAN-ID:20250309-001"
    q1, q2 = generate_qr_image(d, box_size=5), generate_qr_image(d2, box_size=5)
    place_qr(c, q1, 80, 450)
    place_qr(c, q2, A4_W - q2.shape[1] - 80, 900)
    draw_dummy_text_lines(c, 450 + q1.shape[0] + 30, 880)
    draw_dummy_text_lines(c, 900 + q2.shape[0] + 30, A4_H - 80)
    c = add_desk_background(c)
    return [("F04_multi_desk_persp", perspective_warp(c, 0.08), sorted([d, d2]))]

def s_multi_persp_strong(d, n):
    c = create_blank_a4()
    d2 = "REF:ABC-789"
    q1, q2 = generate_qr_image(d, box_size=5), generate_qr_image(d2, box_size=5)
    place_qr(c, q1, 100, 200)
    place_qr(c, q2, A4_W - q2.shape[1] - 100, 1000)
    draw_dummy_text_lines(c, 200 + q1.shape[0] + 30, 980)
    return [("F05_multi_persp_strong", perspective_warp(c, 0.15), sorted([d, d2]))]


# ============================================================
# 全シナリオ
# ============================================================

ALL_SCENARIOS = [
    # A. 基本配置 (6)
    s_doc_center, s_doc_corner_small, s_doc_top_right, s_doc_with_table, s_on_desk, s_tiny_qr,
    # B. 照明・劣化 (13)
    s_shadow_left, s_shadow_right, s_shadow_diagonal, s_shadow_top,
    s_spotlight, s_spotlight_off_center,
    s_paper_texture, s_stained,
    s_low_contrast_print, s_very_low_contrast,
    s_inverted, s_colored_paper, s_heavy_noise,
    # C. 歪み・変形 (21)
    s_perspective_mild, s_perspective_moderate, s_perspective_strong, s_perspective_extreme,
    s_topdown_20, s_topdown_40, s_topdown_55,
    s_side_20, s_side_40,
    s_barrel, s_barrel_strong,
    s_pincushion,
    s_wave, s_wave_strong,
    s_cylindrical, s_cylindrical_tight,
    s_rotated_5, s_rotated_15, s_rotated_45, s_rotated_90, s_rotated_180,
    # D. 撮影条件 (10)
    s_phone_good, s_phone_medium, s_phone_bad,
    s_motion_blur_light, s_motion_blur_heavy,
    s_jpeg_q15, s_jpeg_q5,
    s_blur_gaussian, s_downscale_30pct, s_downscale_15pct,
    # E. 複合条件 (10)
    s_desk_perspective, s_desk_topdown, s_shadow_perspective,
    s_texture_barrel, s_noise_perspective, s_phone_perspective,
    s_shadow_wave, s_combined_hard, s_combined_extreme, s_cylindrical_shadow_desk,
    # F. 複数QR (5)
    s_multi_two, s_multi_three, s_multi_sizes, s_multi_desk_persp, s_multi_persp_strong,
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    records = []
    for qr_name, qr_data in QR_DATA:
        for scenario_fn in ALL_SCENARIOS:
            results = scenario_fn(qr_data, qr_name)
            for scenario_name, img, expected_list in results:
                filename = f"{qr_name}__{scenario_name}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                imwrite_safe(filepath, img)
                records.append({
                    "filename": filename,
                    "expected": expected_list,
                    "data_name": qr_name,
                    "scenario": scenario_name,
                })

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(records)} test images in {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path} ({len(records)} entries)")

    # カテゴリ別集計
    categories = {}
    for r in records:
        cat = r["scenario"].split("_")[0]
        categories.setdefault(cat, []).append(r["scenario"])
    print(f"\nCategories:")
    for cat in sorted(categories):
        scenarios = sorted(set(categories[cat]))
        print(f"  {cat}: {len(scenarios)} scenarios, {len(categories[cat])} images")


if __name__ == "__main__":
    main()
