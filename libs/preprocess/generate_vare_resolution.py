import os
import matplotlib.image as image
from PIL import Image
IMG_PATH = './001.jpg'
VARE_RESOLUTION = [(300,300),(400,400),(512,512),(1280,720),(1920,1080)]

img = Image.open(IMG_PATH)
#img = image.imread(IMG_PATH)
for index,shape in enumerate(VARE_RESOLUTION):
    img = img.resize(shape)
    img.save('00{}.jpg'.format(index+2))