"""Utilities for the sign-language recognition notebook workflow.

The notebook remains the experiment record. This module extracts the reusable
MediaPipe keypoint, LSTM model, dataset-loading, and webcam inference pieces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


ACTIONS = np.array(["my", "name is", "[NAME]"])
SEQUENCE_LENGTH = 30
KEYPOINT_VECTOR_SIZE = 1662

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detections(image: np.ndarray, model: mp_holistic.Holistic):
    """Run MediaPipe holistic detection on one OpenCV frame."""
    image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image: np.ndarray, results) -> None:
    """Draw face, pose, and hand landmarks on a frame."""
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results) -> np.ndarray:
    """Flatten pose, face, and hand landmarks into one fixed-size vector."""
    pose = (
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    left_hand = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, left_hand, right_hand])


def load_sequences(data_dir: Path, actions: np.ndarray = ACTIONS, sequence_length: int = SEQUENCE_LENGTH):
    """Load pre-extracted keypoint sequences saved as ``.npy`` files."""
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence_dir in sorted((data_dir / action).iterdir(), key=lambda p: int(p.name)):
            window = [np.load(sequence_dir / f"{frame_num}.npy") for frame_num in range(sequence_length)]
            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), to_categorical(labels).astype(int)


def build_model(num_actions: int = len(ACTIONS)) -> Sequential:
    """Build the LSTM classifier used by the notebook."""
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, KEYPOINT_VECTOR_SIZE)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_actions, activation="softmax"))
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def train_model(data_dir: Path, output: Path, epochs: int) -> None:
    """Train a model from exported MediaPipe keypoint arrays."""
    x, y = load_sequences(data_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    model.save(output)


def run_webcam(model_path: Path, threshold: float = 0.5) -> None:
    """Run real-time webcam prediction with a trained Keras model."""
    model = load_model(model_path)
    sequence, sentence, predictions = [], [], []

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detections(frame, holistic)
            draw_styled_landmarks(image, results)

            sequence.append(extract_keypoints(results))
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                result = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction = int(np.argmax(result))
                predictions.append(prediction)
                if np.unique(predictions[-10:])[0] == prediction and result[prediction] > threshold:
                    if not sentence or ACTIONS[prediction] != sentence[-1]:
                        sentence.append(ACTIONS[prediction])
                    sentence = sentence[-5:]

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, " ".join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Sign Language Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run the sign-language recognition model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=Path("MP5_Data"))
    train_parser.add_argument("--output", type=Path, default=Path("models/projsen.h5"))
    train_parser.add_argument("--epochs", type=int, default=2000)

    webcam_parser = subparsers.add_parser("webcam")
    webcam_parser.add_argument("--model", type=Path, default=Path("models/projsen.h5"))
    webcam_parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train_model(args.data_dir, args.output, args.epochs)
    elif args.command == "webcam":
        run_webcam(args.model, args.threshold)
