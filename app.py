from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import cv2
import os

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 15

resnet = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=[224,224,3],
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

def frames_extraction_with_resnet(video_path):
    # Declare a list to store video frames.
    frames_list = []

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))  # ResNet-50 input size

        # Preprocess frame for ResNet-50
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img_rgb = np.expand_dims(img_rgb, axis=0)
        img_processed = preprocess_input(img_rgb)

        # Extract features using ResNet-50
        features = resnet.predict(img_processed)

        # Append the preprocessed frame into the frames list
        frames_list.append(features)

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list

def frames_extraction(video_path):

    # Declare a list to store video frames.
    frames_list = []

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Append the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list

def create_data_to_predict(video_file_path):
    features=[]
    frames = frames_extraction_with_resnet(video_file_path)

    if len(frames) == SEQUENCE_LENGTH:

        features.append(frames)

    features = np.asarray(features)

    return features

def create_data_to_predict_clstm(video_file_path):
    features=[]
    frames = frames_extraction(video_file_path)
    if len(frames) == SEQUENCE_LENGTH:

        features.append(frames)

    features = np.asarray(features)

    print("SHAPE : ",features.shape)
    return features

@app.route("/")
def indexku():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data tensor dari request JSON
    data = request.get_json(force=True)
    input_data = np.array(data['input_data'])  # Ubah JSON menjadi numpy array
    method=data['method']
    
    if method=="DSCLSTM":
        model = tf.keras.models.load_model('DDmodel/saved_model_DSC_clstm_e10_lr001.tf')
    else:
        model = tf.keras.models.load_model('DDmodel/saved_model_clstm_e10_lr001.tf')
    # Lakukan prediksi
    prediction = model.predict(input_data)
    print("hasil prediksi : ",prediction)
    if np.argmax(prediction)==0:
        prediction_result="Mengantuk"
    else:
        prediction_result="Tidak Mengantuk"
    
    # Kirim hasil prediksi kembali ke frontend
    return jsonify({'prediction': prediction_result})

classes_list=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

UPLOAD_FOLDER='uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/predictBisindo', methods=['POST'])
def predictBisindo():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'})
    
    # Mendapatkan file video dari request
    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected video file'})
    
    # Menyimpan video ke folder yang ditentukan
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    data = create_data_to_predict(video_path)
    data = data.reshape(1,SEQUENCE_LENGTH, -1)

    model_ResLSTM=load_model('BisindoModel/ResLSTM.h5')
    predicted=model_ResLSTM.predict(data)
    result = np.argmax(predicted)

    print("Result : ", classes_list[result])

    os.remove(video_path)

    return jsonify({'prediction': classes_list[result]})

@app.route('/predictBisindoCLSTM', methods=['POST'])
def predictBisindoCLSTM():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'})
    
    # Mendapatkan file video dari request
    video1 = request.files['video']

    if video1.filename == '':
        return jsonify({'error': 'No selected video file'})
    
    # Menyimpan video ke folder yang ditentukan
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video1.filename)
    video1.save(video_path)

    data1 = create_data_to_predict_clstm(video_path)


    print("SHAPE : ", data1.shape)

    model_CLSTM=load_model('BisindoModel/CLSTM.h5')
    predicted=model_CLSTM.predict(data1)
    result = np.argmax(predicted)


    os.remove(video_path)

    return jsonify({'prediction': classes_list[result]})


if __name__ == "__main__":
    app.run(debug=True)

