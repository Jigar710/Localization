from ultralytics import YOLO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib

def train():
    model = YOLO('yolov8n.pt')
    df = pd.read_csv('spaceData2.csv')
    print(df)
    roomsObj = []
    objects = set()
    for i in range(len(df)):
        oneRoomObj = set()
        folder_dir = "cap/"+df['folderPath'][i]
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".jpg")):
                print(folder_dir+"/"+images)
                results = model.predict(source=folder_dir+"/"+images,save=False)
                for i in range(len(results[0].boxes.cls)):
                    oneRoomObj.add(results[0].names[int(results[0].boxes.cls[i])])
                    objects.add(results[0].names[int(results[0].boxes.cls[i])])
        roomsObj.append(" ".join(oneRoomObj))
    df['objects'] = roomsObj
    set_as_list = list(objects)
    with open("set_as_list.txt", "w") as file:
        file.write(str(set_as_list))
    for object in objects:
        temp = []
        for i in range(len(df)):
            if object in df['objects'][i]:
                temp.append(1)
            else:
                temp.append(0)
        df[str(object)] = temp
    df = df.drop(['objects','folderPath'],axis=1)
    df.to_csv("solution.csv")
    space_encoder = LabelEncoder()
    encoded_space  =  space_encoder.fit_transform(df['space'])
    df['space'] = encoded_space
    print(df.head(12))
    spaceT_encoder = LabelEncoder()
    encoded_spaceT  =  spaceT_encoder.fit_transform(df['spaceType'])
    df['spaceType'] = encoded_spaceT

    building_encoder = LabelEncoder()
    encoded_building  =  building_encoder.fit_transform(df['building'])
    df['building'] = encoded_building

    joblib.dump(space_encoder, 'space_encoder.pkl')
    joblib.dump(spaceT_encoder, 'spaceT_encoder.pkl')
    joblib.dump(building_encoder, 'building_encoder.pkl')
    df = df.drop(["spaceType","building"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size=0.25, random_state=42)

    model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Adjust input shape for the combined data
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(11, activation='softmax')  # Use softmax for multiclass classification
    ])

# Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    model.save("model.h5")
    test_loss, test_accuracy = model.evaluate(X_train, y_train, verbose=1)
    print(f"Test accuracy: {test_accuracy}")
    predictions = model.predict(X_train)
    y_pred = np.argmax(predictions, axis=-1)

    # print(classification_report(y_train, predicted_labels))
    acc = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    f1 = f1_score(y_train, y_pred, average='weighted')

    acc_series = pd.Series(acc, name='Accuracy')
    precision_series = pd.Series(precision, name='Precision')
    recall_series = pd.Series(recall, name='Recall')
    f1_series = pd.Series(f1, name='F1')

    df = pd.concat([acc_series, precision_series, recall_series, f1_series], axis=1)
    return df

# train()