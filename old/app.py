# 必要なモジュールのインポート
import torch
import xml.etree.ElementTree as ET
from SSDNet import Net
from animal import transform # animal.py から前処理とネットワークの定義を読み込み
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

def visualize_results(input, outputs, threshold):
  img= input.permute(1, 2, 0).numpy()
  image = Image.fromarray((img*255).astype(np.uint8))
  aerial_maritime_labels = ['test']
  scale = torch.Tensor(img.shape[1::-1]).repeat(2)
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype(r'C:\Users\t_nak\Desktop\animal_app\weights\NotoSansCJKjp-Bold.otf', 16)

  for i in range(outputs.size(1)):
    j = 0
    while outputs[0,i,j,0] >= threshold:
      score = outputs[0,i,j,0]
      label_name = aerial_maritime_labels[i-1]
      boxes = (outputs[0,i,j,1:]*scale).cpu().numpy()
      draw.rectangle(boxes, outline='red', width=5)
      #w, h = font.getsize(label_name)
      #w, h = draw.textsize(label_name, font=font)
      text_bbox = draw.textbbox((0, 0), label_name, font=font)
      #w, h = text_bbox.size
      w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
      draw.rectangle([boxes[0], boxes[1], boxes[0]+w, boxes[1]+h], fill='red')
      draw.text((boxes[0], boxes[1]), label_name, font=font, fill='white')
      j+=1
  return image

# モデルの初期化
net = Net(phase='test', num_classes=6).cpu().eval()
# net.load_state_dict(torch.load('C:/Users/t_nak/Desktop/animal_app/src/ssd.pt'))
file_path = os.path.abspath('C:/Users/t_nak/Desktop/animal_app/src/ssd.pt')
net.load_state_dict(torch.load(file_path))

# Webエンドポイント
@app.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        # ログの出力
        logging.info("Root endpoint accessed")
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = transforms.ToTensor()(image).unsqueeze(0)

        # モデルによる予測
        with torch.no_grad():
            prediction = net(image)

        # 予測結果の描画
        result_image = visualize_results(image.squeeze(), prediction, threshold=0.5)

        # PIL Imageをバイトデータに変換してレスポンス
        img_byte_array = io.BytesIO()
        result_image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        # ファイルに保存
        with open("result_image.png", "wb") as f:
            f.write(img_byte_array)
            
        return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")
    except Exception as e:
        logging.exception("Error processing file: %s", e)
        return {"error": str(e)}

# FastAPIサーバーの起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)