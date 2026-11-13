"""
WeChatQRCode用のモデルファイルをダウンロードするスクリプト。
opencv_3rdparty リポジトリから4つのファイルを取得する。
"""

import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "wechat_qrcode")

# GitHub raw URLはリダイレクトされるため、HuggingFaceのミラーを使用
BASE_URL = "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode"

FILES = [
    "detect.prototxt",
    "detect.caffemodel",
    "sr.prototxt",
    "sr.caffemodel",
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename in FILES:
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"  既に存在: {filename}")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"  ダウンロード中: {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"    完了 ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"    失敗: {e}")
            print(f"    手動でダウンロードしてください: {url}")
            if os.path.exists(filepath):
                os.remove(filepath)

    print(f"\nモデルディレクトリ: {MODEL_DIR}")


if __name__ == "__main__":
    main()
