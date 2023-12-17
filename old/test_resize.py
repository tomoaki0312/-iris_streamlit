import requests

url = "http://127.0.0.1:8000/resize_image/"

files = {'file': ('test.jpg', open('test.jpg', 'rb'), 'image/jpeg')}
response = requests.post(url, files=files)

# 画像をバイトデータとして保存
img_bytes = response.content
with open('resized_image.jpg', 'wb') as f:
    f.write(img_bytes)