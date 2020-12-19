import os
import pathlib
import glob
import cv2

# setting config
script_path = os.path.join(os.path.dirname(__file__))
dl_path = f'{script_path}/data/'
cascade_path = f'{script_path}/setting/haarcascades/haarcascade_frontalface_default.xml'
outpath = f'{script_path}/data/face'
file_names = os.listdir(dl_path)
messages = []

for file_name in file_names:
    count = 1
    # file 
    pict_path = f'{dl_path}/{file_name}/*'
    out_path = f'{outpath}/{file_name}'
    os.makedirs(out_path, exist_ok=True)
    # get picture fullpath
    file_paths = glob.glob(pict_path)
    sum_files = len(file_paths)
    # \\ → /
    file_paths = ','.join(file_paths)
    file_paths = file_paths.replace('\\', '/')
    file_paths = file_paths.split(',')

    print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print(f'{file_name} folder process start')
    for file_path in file_paths:
        # get filename
        path = pathlib.Path(file_path)
        filename = path.name
        # read picture
        image = cv2.imread(file_path)
        # grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # face authentication
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=15, minSize=(64, 64))
        # Error
        if len(faces) == 0:
            print(f'index {count} {file_name} Face recognition failure')
        count = count + 1
        # face size cut
        for (xpos, ypos, width, height) in faces:
            face_image = image[ypos:ypos+height, xpos:xpos+width]
            if face_image.shape[0] > 64:
                face_image = cv2.resize(face_image, (64, 64))
            path = pathlib.Path()
            # save
            cv2.imwrite(f'{out_path}/{filename}.jpg', face_image)
    face_sum = len(os.listdir(out_path))
    # result message   
    messages.append(f'{file_name} Success:{sum_files - face_sum} failure:{face_sum}')

for message in messages:
    print(message)