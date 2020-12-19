import sys, os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

# setting path
script_path = os.path.join(os.path.dirname(__file__))
data_folder = f'{script_path}/data'
result_folder = f'{script_path}/result'
model_path = f'{data_folder}/model/model.h5'
cascade_path = f'{script_path}/setting/haarcascades/haarcascade_frontalface_default.xml'
check_image_path = f'{result_folder}/s-12124_main-1.jpg'


def image_check(model, cascade_path, image):
    # image color, gray change
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cascade 
    cascade = cv2.CascadeClassifier(cascade_path)
    face_list = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    if len(face_list) != 0:
        for (xpos, ypos, width, height) in face_list:
            face_image = image[ypos:ypos+height, xpos:xpos+width]
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                print('Please make sure the image size is at least 64.')
                sys.exit(0)

            face_image = cv2.resize(face_image, (64, 64))
            cv2.rectangle(image, (xpos, ypos), (xpos+width, ypos+height), (255, 0, 0), thickness=2)
            # shape add
            face_image = np.expand_dims(face_image, axis=0)
            name, result = detect_who(model, face_image)
            # name text add
            cv2.putText(image, f'{name}: {result}', (xpos, ypos+height+20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    else:
        print('Face is undetectable.')
    
    return image

def detect_who(model, face_image):
    name = ''
    result = model.predict(face_image)
    print(f'Hashimoto Kanna possibility: {result[0][0]*100:.3f}%')
    print(f'Aragaki Yui possibility: {result[0][1]*100:.3f}%')
    name_label = np.argmax(result)

    if name_label == 0:
        name = 'Hashimoto Kanna'
        result = f'{result[0][0]*100:.3f}%'
    elif name_label == 1:
        name = 'Aragaki Yui'
        result = f'{result[0][1]*100:.3f}%'
    return name, result

image = cv2.imread(check_image_path)
if image is None:
    print(f'Unable to load image files. {check_image_path}')
    sys.exit(0)
if not os.path.exists(model_path):
    print(f'The model file does not exist. {model_path}')
    sys.exit(0)
model = keras.models.load_model(model_path)

result_image = image_check(model, cascade_path, image)
plt.imshow(result_image)
plt.savefig(f'{result_folder}/result.jpg')
plt.show()
