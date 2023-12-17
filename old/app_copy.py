# 必要なモジュールのインポート
import torch
import xml.etree.ElementTree as ET
from animal import transform, Net # animal.py から前処理とネットワークの定義を読み込み
# from flask import Flask, request, render_template, redirect
import io
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import base64
import os



from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

app = FastAPI()

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import ImageDraw, ImageFont
import io
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

import logging
from fastapi import FastAPI

# ログの設定
logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = FastAPI()

from PIL import Image

def preprocess_image(image):
    # PyTorchのテンソルをNumPy配列に変換
    image_np = image.cpu().numpy()

    # NumPy配列の次元が (1, 1, 640) の場合、squeeze して (640,) に変換
    if image_np.shape[0] == 1:
        image_np = image_np.squeeze(0)

    # グレースケール画像の場合、RGBに変換
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    
    # NumPy配列を正規化（0-1の範囲にスケーリング）
    image_np = image_np.astype(np.float32) / 255.0
    
    # NumPy配列をPIL Imageに変換
    try:
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    except Exception as e:
        print(f"Error converting to PIL Image: {e}")
        return None

    # その他の前処理（リサイズなど）を適用
    transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    input_tensor = transform(image_pil)
    input_batch = input_tensor.unsqueeze(0)  # バッチ次元を追加
    return input_batch

# 画像処理用の関数
def visualize_results(input, outputs, threshold):
    # 仮の描画関数
    img = input.permute(1, 2, 0).numpy()
    image = Image.fromarray((img * 255).astype(np.uint8))
    # 仮の描画処理
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), "Prediction", font=font, fill="red")
    print(outputs.size())
    return image

# モデルの初期化
net = Net(phase='test', num_classes=6).cpu().eval()

# Webエンドポイント
@app.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        # ログの出力
        logging.info("Root endpoint accessed")
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = transforms.ToTensor()(image).unsqueeze(0)

        # 画像の前処理
        input_batch = preprocess_image(image)

        # モデルによる予測
        with torch.no_grad():
            prediction = net(input_batch)

        # 予測結果の描画
        result_image = visualize_results(image.squeeze(), prediction, threshold=0.5)

        # PIL Imageをバイトデータに変換してレスポンス
        img_byte_array = io.BytesIO()
        result_image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")
    except Exception as e:
        logging.exception("Error processing file: %s", e)
        return {"error": str(e)}

# FastAPIサーバーの起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)