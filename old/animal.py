# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from xml.etree.ElementTree import parse
from PIL import ImageDraw, ImageFont
from SSDNet import Net
transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
net = Net()
import os
import urllib.request



# SSD の重み
#net.load_state_dict(torch.load('C:\Users\t_nak\Desktop\animal_app\weights\ssd_300.pth'))
from SSDNet import conf

new_conf = conf(num_classes=6)
net.conf = new_conf
net.num_classes = 6

