import threading
import time
import cv2
from datetime import datetime
from .model import ModelHandler
from .logging import EmotionLogging


class WebcamHandler:
    """
    Handles webcam capture and prediction using a model handler.
    This class manages the webcam capture in a separate thread, allowing
    for real-time predictions based on the captured frames.
    Attributes:
        model_handler (ModelHandler): An instance of ModelHandler to handle model operations.
        logging_handler (EmotionLogging): An instance of EmotionLogging to log predictions.
        capture_active (bool): Flag indicating if the webcam capture is active.
        capture_thread (threading.Thread): Thread for capturing webcam frames.
    """

    def __init__(self, model_handler: ModelHandler):
        self.model_handler = model_handler
        self.logging_handler = EmotionLogging()
        self.capture_active = False
        self.capture_thread = None

    def start_capture(self):
        if self.capture_thread is None:
            self.capture_active = True
            self.capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self.capture_thread.start()
            print("Webcam capture thread started.")

    def stop_capture(self):
        print("Attempting to stop webcam capture thread...")
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.5)
            if self.capture_thread.is_alive():
                print("Webcam capture thread did not stop in time.")
            else:
                print("Webcam capture thread stopped.")
        self.capture_thread = None

    def _capture_loop(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"{datetime.now()}: Error: Could not open webcam.")
                self.capture_active = False
                return
            print(f"{datetime.now()}: Webcam opened successfully.")

            while self.capture_active:
                ret, frame = cap.read()
                if ret:
                    if self.model_handler.ort_session:
                        preprocessed_frame = self.model_handler.preprocess_image(
                            frame.copy()
                        )
                        if preprocessed_frame is not None:
                            predicted_label, confidence, _ = self.model_handler.predict(
                                preprocessed_frame
                            )
                            if predicted_label is not None:
                                self.logging_handler.add_label(
                                    predicted_label, confidence
                                )

                # Small delay to reduce CPU usage
                for _ in range(10):
                    if not self.capture_active:
                        break
                    time.sleep(0.1)
        except Exception as e:
            print(f"{datetime.now()}: Exception in webcam loop: {e}")
        finally:
            if cap and cap.isOpened():
                cap.release()
            print(
                f"{datetime.now()}: Webcam capture thread finished and webcam released."
            )
            self.capture_active = False
