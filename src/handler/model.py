import os
import cv2
import numpy as np
import onnxruntime
from typing import Optional


class ModelHandler:
    """
    A class to handle the ONNX model for emotion recognition.
    This class is responsible for loading the model, preprocessing images,
    and making predictions.
    Attributes:
        onnx_model_path (str): Path to the ONNX model file.
        class_labels (list): List of emotion class labels.
        input_size (tuple): Size of the input image for the model.
        ort_session (onnxruntime.InferenceSession): ONNX runtime session.
        input_name (str): Name of the input tensor for the model.
        output_name (str): Name of the output tensor for the model.
    """

    def __init__(self):
        self.onnx_model_path = "model.onnx"
        self.class_labels = [
            "anger",
            "contempt",
            "disgust",
            "embarrass",
            "fear",
            "joy",
            "neutral",
            "pride",
            "sadness",
            "surprise",
        ]
        self.input_size = (224, 224)
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.onnx_model_path):
            print(
                f"ONNX Model Error: File not found at '{self.onnx_model_path}'. Inference will be disabled."
            )
            return

        try:
            self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"ONNX model '{self.onnx_model_path}' loaded successfully.")
            print(
                f"Model Input Name: {self.input_name}, Output Name: {self.output_name}"
            )
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.ort_session = None

    def preprocess_image(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.ort_session is None:
            return None

        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, preprocessed_frame: np.ndarray):
        if self.ort_session is None or preprocessed_frame is None:
            return None, None, None

        try:
            ort_inputs = {self.input_name: preprocessed_frame}
            ort_outs = self.ort_session.run([self.output_name], ort_inputs)
            scores = ort_outs[0][0]
            predicted_index = np.argmax(scores)
            confidence = float(scores[predicted_index])
            predicted_label = (
                self.class_labels[predicted_index]
                if 0 <= predicted_index < len(self.class_labels)
                else "Unknown"
            )
            return predicted_label, confidence, predicted_index
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None, None, None
