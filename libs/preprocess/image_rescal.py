import os
import matplotlib.image as mpimg
from scipy import misc
imgs_path = '../../datasets/IDImage/JPGImage'
labels_path = '../../datasets/IDImage/Label'
imgs = os.listdir(imgs_path)
labels = os.listdir(labels_path)
for item in imgs:
    id = item.split('.')[0]
    print(id)
    img_path = os.path.join(imgs_path,item)
    label_path = os.path.join(labels_path,id+'.png')
    img = mpimg.imread(img_path)
    label = mpimg.imread(label_path)
    print(label.shape)
    ih,iw,ic = img.shape
    lh,lw,lc = label.shape
    if not ih == lh and iw == lw:
        print('{} doesn`t match'.format(id))
        continue
    if max(ih,iw) > 1280:
        while max(ih,iw) > 1280:
            ih = int(ih * 0.5)
            iw = int(iw * 0.5)
        img = misc.imresize(img,(ih,iw,ic))
        label = misc.imresize(label,(ih,iw,lc))
        misc.imsave(img_path,img)
        misc.imsave(label_path,label)