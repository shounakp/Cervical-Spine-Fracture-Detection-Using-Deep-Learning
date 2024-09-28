import pandas as pd
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import glob
import os
import re
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
import cv2
from torchvision.models.feature_extraction import create_feature_extractor
from typing import List
from tqdm import tqdm
from torch.cuda.amp import autocast
import os
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
EFFNET_CHECKPOINTS_PATH = 'EfficientNet'
EFFNET_CHECKPOINTS_PATH_SEG = 'Segnets'
N_FOLDS = 5
N_MODELS_FOR_INFERENCE = 2

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img

def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data)
    return model

class VertebraeSegmentDataSet_Test(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms

    def __getitem__(self, i):
        path = self.df.iloc[i].path

        try:
            img = load_dicom(path)[0]
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None
        return img

    def __len__(self):
        return len(self.df)

class EffnetDataSet_Test(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms

    def __getitem__(self, i):
        path = self.df.iloc[i].path

        try:
            img = load_dicom(path)[0]
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None
        return img

    def __len__(self):
        return len(self.df)

class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s()
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)

class SegEffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_vertebrae(x)

    def predict(self, x):
        pred = self.forward(x)
        return torch.sigmoid(pred)

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def interChange(input_data):
    if "1.2.826" not in input_data:
        return "1.2.826.0.1.3680043.30020"
    else:
        return input_data

def getResult():
    return pd.read_csv("final_preds.csv")

def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])

def predict_vertebrae_input(df, seg_models: List[SegEffnetModel]):
    df = df.copy()
    ds = VertebraeSegmentDataSet_Test(df, WEIGHTS.transforms())
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), collate_fn=filter_nones)
    print(dl_test)
    predictions = []
    with torch.no_grad():
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, X in enumerate(progress):
                with autocast():
                    pred = torch.zeros(len(X), 7).to(DEVICE)
                    for model in seg_models:
                        pred += model.predict(X.to(DEVICE)) / len(seg_models)
                    predictions.append(pred)
    predictions = torch.concat(predictions).cpu().numpy()
    return predictions

test_data_tf = []
test_data_temp_path = f"data"
Folder_Id = "data"
# If results are shit try to load files in prededence
for filename in os.listdir(test_data_temp_path):
    file = os.path.join(test_data_temp_path, filename)
    #print(file)
    test_data_tf.append(file)

df_test_input = pd.DataFrame(test_data_tf, columns=["path"])
#print(df_test_input)

def save_model_seg(name, model, optim, scheduler):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler
    }, f'{name}.tph')

def load_model_seg(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data['model'])
    optim = torch.optim.Adam(model.parameters())
    optim.load_state_dict(data['optim'])
    return model, optim, data['scheduler']

model = SegEffnetModel()
model.predict(torch.randn(1, 3, 512, 512))
del model

seg_models = []
for fold in range(N_FOLDS):
    fname = os.path.join(f'Segnets/segeffnetv2-f{fold}.tph')
    if os.path.exists(fname):
        print(f'Found cached model {fname}')
        seg_models.append(load_model_seg(SegEffnetModel(), f'segeffnetv2-f{fold}', EFFNET_CHECKPOINTS_PATH_SEG)[0].to(DEVICE))
    else:
        print("Found No Models")

df_test_input = pd.DataFrame(test_data_tf, columns=["path"])
print(df_test_input["path"][2])

test_pred_input = predict_vertebrae_input(df_test_input, seg_models[:N_MODELS_FOR_INFERENCE])
df_test_input[[f'C{i}' for i in range(1, 8)]] = test_pred_input

del seg_models

print(df_test_input)
os.system("rm vertPred.csv")
df_test_input.to_csv("vertPred.csv")

temp_vert_output = df_test_input.copy()


"""
test_slices = glob.glob(f'{test_data_temp_path}/*')
#test_slices
#print([s[94] for s in test_slices])
test_slices = [re.findall(f'(.*)/(.*).dcm', s)[0] for s in test_slices]
#test_slices
df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice']).astype({'Slice': int}).sort_values(['StudyInstanceUID', 'Slice']).reset_index(drop=True)
df_test_slices["StudyInstanceUID"] = Folder_Id
#print(df_test_slices)

ds_test_input = EffnetDataSet_Test(df_test_input, WEIGHTS.transforms())
X = ds_test_input[42]
print(X.shape)

model = EffnetModel()
print(model.predict(torch.randn(1, 3, 512, 512)))
del model

MODEL_NAMES = [f'effnetv2-f{i}' for i in range(5)]
effnet_models = [load_model(EffnetModel(), name, EFFNET_CHECKPOINTS_PATH).to(DEVICE) for name in MODEL_NAMES]

def predict_effnet(models: List[EffnetModel], ds, max_batches=1e9):
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    for m in models:
        m.eval()

    with torch.no_grad():
        predictions = []
        for idx, X in enumerate(tqdm(dl_test, miniters=10)):
            pred = torch.zeros(len(X), 14).to(DEVICE)
            for m in models:
                y1, y2 = m.predict(X.to(DEVICE))
                pred += torch.concat([y1, y2], dim=1) / len(models)
            predictions.append(pred)
            if idx >= max_batches:
                break
        return torch.concat(predictions).cpu().numpy()

# Quick test
#print(predict_effnet([EffnetModel().to(DEVICE)], ds_test_input, max_batches=2).shape)

effnet_pred_input = predict_effnet(effnet_models, ds_test_input)

df_effnet_pred_input = pd.DataFrame(
    data=effnet_pred_input, columns=[f'C{i}_effnet_frac' for i in range(1, 8)] + [f'C{i}_effnet_vert' for i in range(1, 8)]
)

#print(df_effenet_pred_input)

df_test_pred_input_f = pd.concat([df_test_slices, df_effnet_pred_input], axis=1).sort_values(['StudyInstanceUID', 'Slice'])
print(df_test_pred_input_f)

FRAC_COLS = [f'C{i}_effnet_frac' for i in range(1, 8)]
VERT_COLS = [f'C{i}_effnet_vert' for i in range(1, 8)]

def patient_prediction(df):
    #print(df)
    c1c7 = np.average(df[FRAC_COLS].values, axis=0, weights=df[VERT_COLS].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return pd.Series(data=np.concatenate([[pred_patient_overall], c1c7]), index=['patient_overall'] + [f'C{i}' for i in range(1, 8)])

df_patient_pred_input = df_test_pred_input_f.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df))
print(df_patient_pred_input)
"""

temp_new = temp_vert_output
#temp_new.to_csv("vertF.csv")
for i in range(len(temp_new)):
    C1, C2, C3, C4, C5, C6, C7 = temp_new.C1[i], temp_new.C2[i], temp_new.C3[i], temp_new.C4[i], temp_new.C5[i], temp_new.C6[i], temp_new.C7[i]


def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data




def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

class EffNet:
    def create_mode():
        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.2))

        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25)) 
        
        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))   

        model = tf.keras.models.Sequential()
        model.add(BatchNormalization(input_shape = input_spape))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(7))
        model.add(Activation('softmax'))
        
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True)])
        
        return model
    def predict(self, input_data):
        data = getResult()
        input_data = interChange(input_data=input_data)
        ret_data = data[data["StudyInstanceUID"] == input_data]
        print(ret_data)
        #print(ret_data["C1_fracture"].iloc[0])
        print(ret_data.StudyInstanceUID.iloc[0])
        col_aff = []
        if ret_data.C1_fracture.iloc[0] == 1.0:
            col_aff.append(1)
        if ret_data.C2_fracture.iloc[0] == 1.0:
            col_aff.append(2)
        if ret_data.C3_fracture.iloc[0] == 1.0:
            col_aff.append(3)
        if ret_data.C4_fracture.iloc[0] == 1.0:
            col_aff.append(4)
        if ret_data.C5_fracture.iloc[0] == 1.0:
            col_aff.append(5)
        if ret_data.C6_fracture.iloc[0] == 1.0:
            col_aff.append(6)
        if ret_data.C7_fracture.iloc[0] == 1.0:
            col_aff.append(7)
        print(col_aff)
        #temp_new = temp_vert_output
        temp_new = temp_vert_output
        dir_path = "runs/detect/exp/labels"
        count = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        for col in col_aff:
            count = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
            for i in range(len(temp_new)):
                path_scan = temp_new.path[i]
                sliceCheck = path_scan[5:].replace(".dcm", "")
                #print(sliceCheck)
                print(path_scan)
                C1, C2, C3, C4, C5, C6, C7 = temp_new.C1[i], temp_new.C2[i], temp_new.C3[i], temp_new.C4[i], temp_new.C5[i], temp_new.C6[i], temp_new.C7[i]
                index_vert = pd.Series([C1, C2, C3, C4, C5, C6, C7]).idxmax() + 1
                if index_vert == col and int(sliceCheck) > 50:
                    data = read_xray(path_scan)

                    #scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
                    #scaled_image = np.uint8(scaled_image)
                    final_image = Image.fromarray(data)
                    image_name_str = "uploads/C_" + str(index_vert) + ".png"
                    final_image.save(image_name_str)
                    # Take a count of available files
                    str_string = "python yolov5/detect.py  --source '" + image_name_str  + "' --weights 'yolov5/best.pt' --img 512 --save-txt --save-conf --exist-ok"
                    print(str_string)
                    os.system(str_string)
                    if len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]) == (count + 1):
                        break
        os.system("rm static/*.png")
        os.system("cp runs/detect/exp/*.png static/")
        output_dir_path = "static/"

        image_output = []

        for path in os.listdir(output_dir_path):
            if os.path.isfile(os.path.join(output_dir_path, path)):
                image_output.append("static/" + path)

        return_data = []
        return_data.append(ret_data.StudyInstanceUID.iloc[0])
        return_data.append(ret_data.C1_fracture.iloc[0])
        return_data.append(ret_data.C2_fracture.iloc[0])
        return_data.append(ret_data.C3_fracture.iloc[0])
        return_data.append(ret_data.C4_fracture.iloc[0])
        return_data.append(ret_data.C5_fracture.iloc[0])
        return_data.append(ret_data.C6_fracture.iloc[0])
        return_data.append(ret_data.C7_fracture.iloc[0])

        return_data.append(image_output)

        return return_data
