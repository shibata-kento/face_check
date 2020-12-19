import shutil
import random
import glob
import os

from PIL import Image

# setting path
data_folder = os.path.join(os.path.dirname(__file__)) + '/data'
face_folder = f'{data_folder}/face'
test_folder = f'{data_folder}/test'
train_folder = f'{data_folder}/train'

# setting config
mode = '2'
angle = 15
size = 64

def test_move(test_folder, in_pic, fname):
    random.shuffle(in_pic)
    os.makedirs(f'{test_folder}/{fname}', exist_ok=True)
    print(f"makedir test folder {fname} finished")
    for t in range(len(in_pic)//5):
        shutil.copy(str(in_pic[t]), f'{test_folder}/{fname}/') 

def inflation_mode(train_folder, in_pic, fname):
    os.makedirs(f'{train_folder}/{fname}', exist_ok=True)
    print(f"makedir train folder {fname} finished")
    for pic in in_pic:
        img = Image.open(pic) 
        img.resize((size, size), Image.LANCZOS)
        img.save(pic)
        shutil.copy(pic, f'{train_folder}/{fname}/')
        for i in range(1, 3):
            img = img.rotate(angle*i)
            pic = pic.replace('.jpg', '')
            img.save(f'{pic}_{str(angle*i)}_rotation.jpg')
            shutil.move(f'{pic}_{str(angle*i)}_rotation.jpg', f'{train_folder}/{fname}/')

# folder name get
file_list = []
for folder in os.listdir(face_folder):
    file_list.append(folder)

# train_test_split
for fname in file_list:
    in_data = f'{face_folder}/{fname}/*'
    in_pic = glob.glob(in_data)
    img_file_name_list = os.listdir(data_folder)
    hoge = ','.join(in_pic)
    hoge = hoge.replace('\\', '/')
    in_pic = hoge.split(',')

    if mode == '1':
        os.makedirs(test_folder, exist_ok=True)
        test_move(test_folder, in_pic, fname)
        print(f"{fname} process end.")
    elif mode == '2':
        os.makedirs(train_folder, exist_ok=True)
        inflation_mode(train_folder, in_pic, fname)
        print(f"{fname} process end.")
