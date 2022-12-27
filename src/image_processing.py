import os
import sys
import shutil
from PIL import Image

raw_path = './dataset/malimg_dataset/malimg_paper_dataset_imgs/'


if not os.path.isdir('./classed_dataset'):
    os.mkdir('./classed_dataset')
if not os.path.isdir('./classed_dataset/trainning_data/'):
    os.mkdir('./classed_dataset/trainning_data/')
if not os.path.isdir('./classed_dataset/test_data/'):
    os.mkdir('./classed_dataset/test_data/')


for i in os.listdir(raw_path):
    if not os.path.isdir(raw_path+i):
        continue
    imgs = os.listdir(raw_path+i)
    if not os.path.exists('./classed_dataset/trainning_data/'+i) \
    and not os.path.exists('./classed_dataset/test_data/'+i):
        os.mkdir('./classed_dataset/trainning_data/'+i)
        os.mkdir('./classed_dataset/test_data/'+i)
    else:
        continue

    for j in range(len(imgs)):
        src_path = f'{raw_path}{i}/{imgs[j]}'
        dst_path = [f'./classed_dataset/trainning_data/{i}/{imgs[j]}',
        f'./classed_dataset/test_data/{i}/{imgs[j]}']

        img = Image.open(src_path)
        img = img.resize((64,64), Image.Resampling.LANCZOS )

        if j > len(imgs)*3/10:
            img.save(dst_path[0])
        else:
            img.save(dst_path[1])
    