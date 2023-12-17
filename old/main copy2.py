from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from torchvision import transforms
from SSDNet import Net
from io import BytesIO
import torch
import torchvision
from torchvision import transforms
from glob import glob
import numpy as np
import base64
import os
import asyncio
from PIL import ImageDraw, ImageFont, Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='test'):
        self.root = root
        self.mode = mode
        self.aerial_maritime_labels = ['dennshinnbashira']
        self.transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
        self.image_list = sorted(glob(f'{root}\{mode}\*.jpg'))
        self.targets = [
            {'bbox': [0, 0, 0, 0], 'label': 0} 
            for _ in range(len(self.image_list))
        ]

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        target = self.targets[idx]

        targets = torch.tensor([target['bbox'][0], target['bbox'][1], target['bbox'][2], target['bbox'][3], target['label']], dtype=torch.float32)
        return img, targets

    def __len__(self):
        return len(self.image_list)

def visualize_results(input, outputs, threshold):
  img= input.permute(1, 2, 0).numpy()
  image = Image.fromarray((img*255).astype(np.uint8))
  print(f"Image shape: {image.size}")
  aerial_maritime_labels = ['dennshinnbashira']
  print(f"Image shape: {image.size}")
  scale = torch.Tensor(img.shape[1::-1]).repeat(2)
  draw = ImageDraw.Draw(image)
  base_dir = os.path.abspath(os.path.dirname(__file__))
  font_path = os.path.join(base_dir, 'NotoSansCJKjp-Bold.otf')
  font = ImageFont.truetype(font_path, 16)

  for i in range(outputs.size(1)):
    j = 0
    while j < outputs.size(2) and outputs[0, i, j, 0] >= threshold:
      score = outputs[0,i,j,0]
      label_name = aerial_maritime_labels[i-1]
      boxes = (outputs[0,i,j,1:]*scale).cpu().numpy()
      draw.rectangle(boxes, outline='red', width=5)
      text_bbox = draw.textbbox((0, 0), label_name, font=font)
      w, h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
      draw.rectangle([boxes[0], boxes[1], boxes[0]+w, boxes[1]+h], fill='red')
      draw.text((boxes[0], boxes[1]), label_name, font=font, fill='white')
      j+=1
  return image

net = Net(phase='test', num_classes=6).cpu().eval()
file_path = 'ssd.pt'
net.load_state_dict(torch.load(file_path))

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # templatesディレクトリにindex.htmlがあることを確認
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/resize_image/")
async def resize_image(request: Request,
    file: UploadFile = File(...),
):
    
    file_contents = file.file.read()

    # ファイル名を変更
    new_filename = "test.jpg"

    # 受け取ったファイルを保存
    upload_path = os.path.abspath("uploads")
    os.makedirs(upload_path, exist_ok=True)
    saved_file_path = os.path.join(upload_path, new_filename)

    # データセットの取得
    root = os.path.abspath(os.path.dirname(__file__))
    test = Dataset(root, mode='uploads')

    with open(saved_file_path, "wb") as f:
        f.write(file_contents)

    image = None  # 初期化

    for n in range(len(test)):
       x, t = test[n]
       y = net(x.unsqueeze(0))
       image = visualize_results(x, y, threshold=0.5)

    # 画像サイズを変更
    if image is not None:
        resized_img = image.resize((600, 600))
        # 画像サイズを確認
        print(f"Resized image shape: {resized_img.size}")

        # 変更された画像をバイトデータに変換
        buffered = BytesIO()
        resized_img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
    
        # 保存したファイルを削除（オプション）
        os.remove(saved_file_path)

        # キャッシュを無効にするためのヘッダーを追加
        headers = {
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
        }

        # 変更された画像をStreamingResponseとして返す
        return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg", headers=headers)
    else:
        print("No image available.")
        # 何かのエラーレスポンスを返すか、リダイレクトなどの対応を行う
        return JSONResponse(content={"error": "No image available."}, status_code=500)