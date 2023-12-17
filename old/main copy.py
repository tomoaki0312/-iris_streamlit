from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import xml.etree.ElementTree as ET
from torchvision import transforms
from SSDNet import Net
from io import BytesIO
import torch
import torchvision
from torchvision import transforms
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import base64
import os
from PIL import ImageDraw, ImageFont, Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='test'):
        self.root = root
        self.mode = mode
        self.aerial_maritime_labels =  ['dennshinnbashira']
        self.transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
        self.data_list = sorted(glob(f'{root}\{mode}\*xml'))
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        parser = ET.parse(data)
        #parser = parse(data)
        img_path = parser.find('filename').text
        img_path = f'{self.root}\{self.mode}\{img_path}'
        img = Image.open(img_path)
        img = self.transform(img)
        target = []
        for obj in parser.findall('object'):
           bbox = obj.find('bndbox')
           xmin = 0
           ymin = 0
           xmax = 0
           ymax = 0
           name = obj.find('name').text
           label = int(self.aerial_maritime_labels.index(name))
           target.append([xmin, ymin, xmax, ymax, label])
        targets = torch.tensor(target, dtype=torch.float32)
        return img, targets
    def __len__(self):
        return len(self.data_list)
# データセットの取得
#root = r'C:\Users\t_nak\Desktop\animal_app'
root = os.path.abspath(os.path.dirname(__file__))
#root = os.path.join(base_dir, 'src')
test = Dataset(root, mode='src')

def visualize_results(input, outputs, threshold):
  img= input.permute(1, 2, 0).numpy()
  image = Image.fromarray((img*255).astype(np.uint8))
  aerial_maritime_labels = ['dennshinnbashira']
  scale = torch.Tensor(img.shape[1::-1]).repeat(2)
  draw = ImageDraw.Draw(image)
#  font = ImageFont.truetype(r'C:\Users\t_nak\Desktop\animal_app\src\NotoSansCJKjp-Bold.otf', 16)
  base_dir = os.path.abspath(os.path.dirname(__file__))
  font_path = os.path.join(base_dir, 'NotoSansCJKjp-Bold.otf')
  font = ImageFont.truetype(font_path, 16)

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

net = Net(phase='test', num_classes=6).cpu().eval()
file_path = 'ssd.pt'
net.load_state_dict(torch.load(file_path))



app = FastAPI()
templates = Jinja2Templates(directory="templates")  # templatesディレクトリにindex.htmlがあることを確認
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/resize_image/")
#async def resize_image(file: UploadFile = File(...)):
    # 受け取ったファイルの内容を確認するためにログに出力
async def resize_image(request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(...),  # フォームからしきい値を取得する例
):
    file_contents = file.file.read()
    print(f"File contents: {file_contents[:100]}...")  # 先頭の一部分を表示

    # ファイル名を確認するためにログに出力
    print(f"Received file: {file.filename}")

    # 受け取ったファイルを保存
    upload_path = os.path.abspath("uploads")
    os.makedirs(upload_path, exist_ok=True)
    saved_file_path = os.path.join(upload_path, file.filename)

    with open(saved_file_path, "wb") as f:
        f.write(file_contents)

    for n in range(len(test)):
       x, t = test[n]
       y = net(x.unsqueeze(0))
       image = visualize_results(x, y, threshold=0.5)

    # 保存したファイルをPIL Imageに変換
    #img = Image.open(saved_file_path)

    # 画像サイズを変更
    resized_img = image.resize((300, 300))  # 例として300x300に変更

    # 変更された画像をバイトデータに変換
    buffered = BytesIO()
    resized_img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    # 保存したファイルを削除（オプション）
    #os.remove(saved_file_path)

    # 変更された画像をStreamingResponseとして返す
    return StreamingResponse(BytesIO(img_bytes), media_type="image/jpeg")
    #return templates.TemplateResponse("index.html", {"request": request})