DATA_PATH = '/User/demos/demos/faces/dataset/'
ARTIFACTS_PATH = '/User/demos/demos/faces/artifacts/'
MODELS_PATH = '/User/demos/demos/faces/models.py'

import torch
import horovod.torch as hvd
import os
import shutil
import zipfile
from urllib.request import urlopen
from io import BytesIO
import face_recognition
from imutils import paths
import cv2
from mlrun.artifacts import TableArtifact
import pandas as pd
import datetime
import random
import string
import v3io_frames as v3f


def encode_images(context, cuda=False, horovod=False):
    if not horovod or hvd.rank() == 0:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        context.logger.info(f'Running on device: {device}')

        client = v3f.Client("framesd:8081", container="users")

        if not os.path.exists(DATA_PATH + 'processed'):
            os.makedirs(DATA_PATH + 'processed')

        if not os.path.exists(DATA_PATH + 'label_pending'):
            os.makedirs(DATA_PATH + 'label_pending')

        if not os.path.exists(DATA_PATH + 'input'):
            os.makedirs(DATA_PATH + 'input')
            resp = urlopen('https://iguazio-public.s3.amazonaws.com/roy-actresses/Actresses.zip')
            zip_ref = zipfile.ZipFile(BytesIO(resp.read()), 'r')
            zip_ref.extractall(DATA_PATH + 'input')
            zip_ref.close()

        if os.path.exists(DATA_PATH + 'input/__MACOSX'):
            shutil.rmtree(DATA_PATH + 'input/__MACOSX')

        idx_file_path = ARTIFACTS_PATH+"idx2name.csv"
        if os.path.exists(idx_file_path):
            idx2name_df = pd.read_csv(idx_file_path)
        else:
            idx2name_df = pd.DataFrame(columns=['value', 'name'])

        new_classes_names = [f for f in os.listdir(DATA_PATH + 'input') if not '.ipynb' in f and f not in idx2name_df['name'].values]

        initial_len = len(idx2name_df)
        final_len = len(idx2name_df) + len(new_classes_names)
        for i in range(initial_len, final_len):
            idx2name_df.loc[i] = {'value': i, 'name': new_classes_names.pop()}

        name2idx = idx2name_df.set_index('name')['value'].to_dict()

        context.log_artifact(TableArtifact('idx2name', df=idx2name_df), target_path='idx2name.csv')

        imagePaths = [f for f in paths.list_images(DATA_PATH + 'input') if not '.ipynb' in f]
        knownEncodings = []
        knownLabels = []
        fileNames = []
        urls = []
        for (i, imagePath) in enumerate(imagePaths):
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            file_name = imagePath.split(os.path.sep)[-1]
            new_path = DATA_PATH + 'processed/' + file_name

            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model='hog')

            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                file_name = name + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))                                                           
                knownEncodings.append(enc)
                knownLabels.append([name2idx[name]])
                fileNames.append(file_name)
                urls.append(new_path)

            shutil.move(imagePath, new_path)

        df_x = pd.DataFrame(knownEncodings, columns=['c' + str(i).zfill(3) for i in range(128)]).reset_index(drop=True)
        df_y = pd.DataFrame(knownLabels, columns=['label']).reset_index(drop=True)
        df_details = pd.DataFrame([['initial training']*3]*len(df_x), columns=['imgUrl', 'camera', 'time'])
        df_details['time'] = [datetime.datetime.utcnow()]*len(df_x)
        df_details['imgUrl'] = urls
        data_df = pd.concat([df_x, df_y, df_details], axis=1)
        data_df['fileName'] = fileNames

        client.write(backend='kv', table='iguazio/demos/demos/faces/artifacts/encodings', dfs=data_df, index_cols=['fileName'])

        with open('encodings_path.txt', 'w+') as f:
            f.write('iguazio/demos/demos/faces/artifacts/encodings')
        context.log_artifact('encodings_path', src_path=f.name, target_path=f.name)
        os.remove('encodings_path.txt')