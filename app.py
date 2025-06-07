from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import tempfile
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
import mediapipe as mp
import traceback

# Initialize mediapipe face_mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load model
model = tf.keras.models.load_model("best_multi_model3_acc99.h5")

# Print model input names → very important to check
print("Model inputs:", model.inputs)


# Functions from your notebook
def mediapipe_predictions(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_frame)
    return results


def extract_frame_points(face_landmarks):
    frame_points = []
    for landmark in face_landmarks.landmark:
        frame_points.extend([landmark.x, landmark.y, landmark.z])
    return frame_points


def extract_facial_features_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    video_segments = []
    current_segment = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = mediapipe_predictions(frame, face_mesh)

        if results.multi_face_landmarks:
            landmarks = extract_frame_points(results.multi_face_landmarks[0])
        else:
            landmarks = np.zeros(478 * 3)

        current_segment.append(landmarks)

        if len(current_segment) == 32:
            video_segments.append(np.array(current_segment))
            current_segment = []

    video_capture.release()

    if 0 < len(current_segment) < 32:
        while len(current_segment) < 32:
            current_segment.append(np.zeros(478 * 3))
        video_segments.append(np.array(current_segment))

    return np.array(video_segments)


def extract_audio_from_video(video_path, temp_audio_path="temp_audio.wav"):
    with VideoFileClip(video_path) as video:
        if video.audio:
            video.audio.write_audiofile(temp_audio_path, codec="pcm_s16le")
        else:
            raise Exception("No audio found in video")
    return temp_audio_path


def extract_mfcc_segments(audio_path, segment_duration=1.07, sr=16000, n_mfcc=32):
    audio, sr = librosa.load(audio_path, sr=sr)
    segment_length = int(segment_duration * sr)
    mfcc_segments = []

    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_segments.append(mfcc_mean)

    return np.array(mfcc_segments).reshape(-1, n_mfcc)


# Flask app
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        print("Extracting facial features...")
        facial_segments = extract_facial_features_from_video(video_path)

        print("Extracting audio and MFCC features...")
        temp_audio_path = extract_audio_from_video(video_path)
        mfcc_segments = extract_mfcc_segments(temp_audio_path)
        os.remove(temp_audio_path)

        min_segments = min(len(facial_segments), len(mfcc_segments))
        facial_segments = facial_segments[:min_segments]
        mfcc_segments = mfcc_segments[:min_segments]

        print(f"Total segments: {min_segments}")
        predictions = []

        for i in range(min_segments):
            facial_input = facial_segments[i].reshape(1, 32, 1434)
            voice_input = mfcc_segments[i].reshape(1, 32)

            pred = model.predict(
                {"facial_input": facial_input, "voice_input": voice_input}
            )

            predictions.append(pred[0])

        predictions = np.array(predictions)
        avg_prediction = np.mean(predictions, axis=0)
        final_label = np.argmax(avg_prediction)
        final_conf = np.max(avg_prediction)
        result = "Lie" if final_label == 1 else "Truth"

        os.remove(video_path)

        return jsonify({"result": result, "confidence": float(final_conf)})

    except Exception as e:
        traceback.print_exc()  # Show full error in terminal
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
