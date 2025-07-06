import sys
import json
import threading
import time
from typing import Literal
import cv2
import numpy as np
import onnxruntime
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QScreen
from datetime import datetime
import os


class ModernMentalHealthSurveyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental Health Check-in (Open Questions + Camera Log)")

        # --- Define Log Directory ---
        self.log_directory = "logs"
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            print(
                f"Logs will be saved in directory: '{os.path.abspath(self.log_directory)}'"
            )
        except OSError as e:
            print(
                f"Error creating log directory '{self.log_directory}': {e}. Logs will be saved in current directory."
            )
            self.log_directory = "."

        # --- ONNX Model Configuration ---
        # PASTIKAN FILE MODEL.ONNX ADA DI DIREKTORI YANG SAMA
        # ATAU UBAH PATH SESUAI LOKASI MODEL ANDA.
        # --- ONNX Model Configuration (USER ACTION REQUIRED) ---
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
        self.input_size = (224, 224)  # Expected input H, W for the model
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self._load_onnx_model()
        # --- End ONNX Configuration ---

        self.questions = [
            {
                "text": "1. Bagaimana kabarmu belakangan ini, ada hal yang membuatmu merasa sangat senang atau sedih?",
            },
            {
                "text": "2. Akhir-akhir ini, apa saja hal yang paling menyita pikiranmu, baik itu dalam pekerjaan, hubungan, atau hal lainnya?",
            },
            {
                "text": "3. Adakah perubahan signifikan yang kamu rasakan dalam dirimu, seperti pola tidur, nafsu makan, atau energimu?",
            },
            {
                "text": "4. Dalam beberapa waktu terakhir, apakah ada momen di mana kamu merasa sangat kewalahan, putus asa, atau bahkan tidak berharga? Jika iya, bisakah kamu ceritakan lebih banyak?",
            },
            {
                "text": "5. Jika kamu bisa mengubah satu hal yang berkaitan dengan perasaanmu saat ini, apa itu, dan apa harapanmu untuk dirimu sendiri ke depannya?",
            },
        ]
        self.num_questions = len(self.questions)
        self.current_question_index = 0
        self.user_answers = [""] * self.num_questions

        base_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.survey_log_file_name = os.path.join(
            self.log_directory, f"survey_log_{base_timestamp_str}.json"
        )
        self.prediction_log_file_name = os.path.join(
            self.log_directory, f"prediction_log_{base_timestamp_str}.json"
        )
        self.answer_log_file_name = os.path.join(
            self.log_directory, f"open_answers_log_{base_timestamp_str}.json"
        )

        print(f"Current session survey log file: {self.survey_log_file_name}")
        print(f"Current session prediction log file: {self.prediction_log_file_name}")
        print(f"Current session open answers log file: {self.answer_log_file_name}")

        self._log_event(
            action_type="passive", event_type="app_init"
        )

        self.capture_active = False
        self.capture_thread = None

        self.init_ui()
        self.apply_styles()
        self._center_window()

        self._start_webcam_capture()
        self.display_question()
        self._log_event(
            action_type="passive",
            event_type="question_displayed",
            details={"question_index": self.current_question_index + 1},
        )

    def _load_onnx_model(self):
        # ... (same as before) ...
        if not os.path.exists(self.onnx_model_path):
            print(
                f"ONNX Model Error: File not found at '{self.onnx_model_path}'. Inference will be disabled."
            )
            self.ort_session = None
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

    def _center_window(self):
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            center_point = screen_geometry.center()
            self.move(
                center_point.x() - self.width() // 2,
                center_point.y() - self.height() // 2,
            )

    def _log_event(
        self, action_type: Literal["active", "passive"], event_type: str, details=None
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "event_type": event_type,
        }
        if details:
            log_entry["details"] = details

        all_logs = []
        try:
            if (
                os.path.exists(self.survey_log_file_name)
                and os.path.getsize(self.survey_log_file_name) > 0
            ):
                with open(self.survey_log_file_name, "r", encoding="utf-8") as f:
                    all_logs = json.load(f)
                if not isinstance(all_logs, list):
                    all_logs = []
            else:
                all_logs = []
        except json.JSONDecodeError:
            all_logs = []
        except Exception:
            all_logs = []

        all_logs.append(log_entry)
        try:
            with open(self.survey_log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(
                f"Error writing to survey log file '{self.survey_log_file_name}': {e}"
            )

    def _log_open_answer(self, question_text: str, answer_text: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "question": question_text,
            "answer": answer_text,
            "question_index": self.current_question_index + 1
        }

        all_answers = []
        try:
            if (
                os.path.exists(self.answer_log_file_name)
                and os.path.getsize(self.answer_log_file_name) > 0
            ):
                with open(self.answer_log_file_name, "r", encoding="utf-8") as f:
                    all_answers = json.load(f)
                if not isinstance(all_answers, list):
                    all_answers = []
            else:
                all_answers = []
        except json.JSONDecodeError:
            all_answers = []
        except Exception:
            all_answers = []

        all_answers.append(log_entry)
        try:
            with open(self.answer_log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_answers, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to open answers log file '{self.answer_log_file_name}': {e}")

    def _log_image_prediction(
        self, predicted_label: str, confidence: float, predicted_index: int
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "predicted_index": int(predicted_index),
        }

        all_predictions = []
        try:
            if (
                os.path.exists(self.prediction_log_file_name)
                and os.path.getsize(self.prediction_log_file_name) > 0
            ):
                with open(self.prediction_log_file_name, "r", encoding="utf-8") as f:
                    all_predictions = json.load(f)
                if not isinstance(all_predictions, list):
                    print(
                        f"Warning: Prediction log file '{self.prediction_log_file_name}' was not a list. Resetting."
                    )
                    all_predictions = []
            else:
                all_predictions = []
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from '{self.prediction_log_file_name}'. Starting with a new prediction log list."
            )
            all_predictions = []
        except Exception as e:
            print(
                f"Error reading prediction log file '{self.prediction_log_file_name}': {e}. Starting with a new list."
            )
            all_predictions = []

        all_predictions.append(log_entry)

        try:
            with open(self.prediction_log_file_name, "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(
                f"Error writing to prediction log file '{self.prediction_log_file_name}': {e}"
            )

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _start_webcam_capture(self):
        if self.capture_thread is None:
            self.capture_active = True
            self.capture_thread = threading.Thread(
                target=self._webcam_capture_loop, daemon=True
            )
            self.capture_thread.start()
            print("Webcam capture thread started.")

    def _webcam_capture_loop(self):
        # ... (modified to call _log_image_prediction) ...
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
                    if self.ort_session:
                        try:
                            preprocessed_frame = self._preprocess_image(frame.copy())
                            ort_inputs = {self.input_name: preprocessed_frame}
                            ort_outs = self.ort_session.run(
                                [self.output_name], ort_inputs
                            )
                            scores = ort_outs[0][0]
                            predicted_index = np.argmax(scores)
                            confidence = float(scores[predicted_index])
                            predicted_label = "Unknown"
                            if 0 <= predicted_index < len(self.class_labels):
                                predicted_label = self.class_labels[predicted_index]

                            # Log to the dedicated prediction log file
                            self._log_image_prediction(
                                predicted_label, confidence, predicted_index
                            )

                        except Exception as e:
                            print(
                                f"{datetime.now()}: Error during model inference: {e}"
                            )
                else:
                    print(
                        f"{datetime.now()}: Error: Failed to capture frame from webcam."
                    )

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

    def _stop_webcam_capture(self):
        print("Attempting to stop webcam capture thread...")
        self.capture_active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.5)
            if self.capture_thread.is_alive():
                print("Webcam capture thread did not stop in time.")
            else:
                print("Webcam capture thread stopped.")
        self.capture_thread = None

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        self.disclaimer_label = QLabel(
            "<b>Penting:</b> Ini BUKAN alat diagnostik. Konsultasikan dengan profesional untuk masalah kesehatan mental."
        )
        self.disclaimer_label.setObjectName("DisclaimerLabel")
        self.disclaimer_label.setWordWrap(True)
        self.main_layout.addWidget(self.disclaimer_label)

        self.instructions_label = QLabel(
            "Silakan jawab pertanyaan-pertanyaan berikut berdasarkan perasaan Anda selama <b>beberapa waktu terakhir</b>."
        )
        self.instructions_label.setObjectName("InstructionsLabel")
        self.instructions_label.setWordWrap(True)
        self.main_layout.addWidget(self.instructions_label)

        self.question_area_widget = QWidget()
        self.question_area_widget.setObjectName("QuestionAreaWidget")
        question_area_layout = QVBoxLayout(self.question_area_widget)
        question_area_layout.setContentsMargins(20, 15, 20, 15)
        question_area_layout.setSpacing(15)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("ProgressLabel")
        question_area_layout.addWidget(
            self.progress_label, alignment=Qt.AlignmentFlag.AlignRight
        )

        self.question_label = QLabel("Teks pertanyaan akan muncul di sini.")
        self.question_label.setObjectName("QuestionLabel")
        self.question_label.setWordWrap(True)
        question_area_layout.addWidget(self.question_label)

        self.answer_input = QTextEdit()
        self.answer_input.setObjectName("AnswerInput")
        self.answer_input.setPlaceholderText("Ketik jawaban Anda di sini...")
        self.answer_input.setFixedHeight(100)
        question_area_layout.addWidget(self.answer_input)

        question_area_layout.addSpacerItem(
            QSpacerItem(5, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        self.main_layout.addWidget(self.question_area_widget)

        self.nav_buttons_layout = QHBoxLayout()
        self.nav_buttons_layout.setSpacing(10)

        self.prev_button = QPushButton("Sebelumnya")
        self.prev_button.setObjectName("PrevButton")
        self.prev_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_button.clicked.connect(self.go_previous)
        self.nav_buttons_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        self.nav_buttons_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Selanjutnya")
        self.next_button.setObjectName("NextButton")
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.go_next)
        self.nav_buttons_layout.addWidget(self.next_button)

        self.main_layout.addLayout(self.nav_buttons_layout)

        self.setMinimumSize(550, 500)
        self.resize(600, 550)

    def apply_styles(self):
        qss = """
            QWidget#MainWindow { background-color: #F4F6F8; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel#DisclaimerLabel { background-color: #FFF3CD; color: #664D03; border: 1px solid #FFECB5; border-radius: 6px; padding: 12px; font-size: 9pt; font-weight: normal; }
            QLabel#InstructionsLabel { color: #4A5568; font-size: 11pt; padding-left: 5px; margin-bottom: 0px; }
            QWidget#QuestionAreaWidget { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E2E8F0; }
            QLabel#QuestionLabel { color: #1A202C; font-size: 13pt; font-weight: bold; line-height: 1.5; padding-bottom: 10px; }
            QLabel#ProgressLabel { color: #718096; font-size: 9pt; font-weight: bold; }
            QTextEdit#AnswerInput {
                border: 1px solid #CBD5E0;
                border-radius: 5px;
                padding: 10px;
                font-size: 10pt;
                color: #2D3748;
                background-color: #FFFFFF;
            }
            QTextEdit#AnswerInput:focus {
                border: 1px solid #3182CE;
            }
            QPushButton { font-size: 11pt; font-weight: bold; padding: 10px 20px; border-radius: 6px; border: none; min-width: 100px; }
            QPushButton#NextButton { background-color: #3182CE; color: white; }
            QPushButton#NextButton:hover { background-color: #2B6CB0; }
            QPushButton#NextButton:pressed { background-color: #2C5282; }
            QPushButton#PrevButton { background-color: #E2E8F0; color: #2D3748; }
            QPushButton#PrevButton:hover { background-color: #CBD5E0; }
            QPushButton#PrevButton:pressed { background-color: #A0AEC0; }
            QPushButton:disabled { background-color: #E2E8F0; color: #A0AEC0; }
            QMessageBox { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }
            QMessageBox QLabel { color: #2D3748; }
            QMessageBox QPushButton { background-color: #CBD5E0; color: #2D3748; padding: 8px 15px; border-radius: 4px; min-width: 80px; }
            QMessageBox QPushButton:hover { background-color: #A0AEC0; }
        """
        self.setStyleSheet(qss)

    def display_question(self):
        # Save answer from previous question before displaying new one
        # This logic ensures the answer from the *previous* question is logged before moving to the *current* one
        if self.current_question_index < self.num_questions and self.current_question_index >= 0:
            if self.current_question_index > 0: # If not on the very first question
                # The answer for the question we are *leaving* (current_question_index - 1)
                # would have been saved in self.user_answers already by go_next/go_previous.
                # Here we just ensure it's logged if it wasn't due to direct navigation or initial display.
                # We can refine this to avoid double-logging if go_next/go_previous already logged it.
                # For simplicity, we'll log it directly in go_next/go_previous.
                pass # Logic moved to go_next/go_previous for more precise logging point

        if self.current_question_index < self.num_questions:
            question_data = self.questions[self.current_question_index]
            self.question_label.setText(question_data["text"])
            self.progress_label.setText(
                f"Pertanyaan {self.current_question_index + 1} dari {self.num_questions}"
            )

            self.answer_input.setText(self.user_answers[self.current_question_index])

            self.prev_button.setEnabled(self.current_question_index > 0)
            if self.current_question_index == self.num_questions - 1:
                self.next_button.setText("Selesai")
            else:
                self.next_button.setText("Selanjutnya")
        else:
            self.process_survey()

    def go_next(self):
        current_q_idx_for_log = self.current_question_index + 1
        action_text = (
            "next_clicked" if self.next_button.text() == "Selanjutnya" else "finish_clicked"
        )

        current_answer = self.answer_input.toPlainText().strip()
        self.user_answers[self.current_question_index] = current_answer

        if not current_answer:
            QMessageBox.warning(
                self, "Jawaban Kosong", "Silakan isi jawaban Anda sebelum melanjutkan."
            )
            self._log_event(
                action_type="passive",
                event_type="validation_error",
                details={
                    "message": "Tidak ada jawaban yang diisi untuk pertanyaan "
                    + str(current_q_idx_for_log)
                },
            )
            return

        # Log the answer to the dedicated file when moving *from* this question
        self._log_open_answer(self.questions[self.current_question_index]["text"], current_answer)

        self._log_event(
            action_type="active",
            event_type=action_text,
            details={"from_question": current_q_idx_for_log, "answer_preview": current_answer[:50] + "..." if len(current_answer) > 50 else current_answer},
        )


        if self.current_question_index < self.num_questions - 1:
            self.current_question_index += 1
            self.display_question()
            self._log_event(
                action_type="passive",
                event_type="question_displayed",
                details={"question_index": self.current_question_index + 1},
            )
        else:
            self.process_survey()

    def go_previous(self):
        current_q_idx_for_log = self.current_question_index + 1
        # Save current answer before moving
        current_answer = self.answer_input.toPlainText().strip()
        self.user_answers[self.current_question_index] = current_answer

        # Log the answer to the dedicated file when moving *from* this question
        self._log_open_answer(self.questions[self.current_question_index]["text"], current_answer)

        self._log_event(
            action_type="active",
            event_type="previous_clicked",
            details={"from_question": current_q_idx_for_log, "answer_preview": current_answer[:50] + "..." if len(current_answer) > 50 else current_answer},
        )
        if self.current_question_index > 0:
            self.current_question_index -= 1
            self.display_question()
            self._log_event(
                action_type="passive",
                event_type="question_displayed",
                details={"question_index": self.current_question_index + 1},
            )

    def process_survey(self):
        # Ensure the last answer is saved and logged before processing
        current_answer = self.answer_input.toPlainText().strip()
        self.user_answers[self.current_question_index] = current_answer
        self._log_open_answer(self.questions[self.current_question_index]["text"], current_answer)


        self._log_event(action_type="passive", event_type="survey_submitted")

        if any(not answer for answer in self.user_answers):
            QMessageBox.warning(
                self,
                "Survei Belum Lengkap",
                "Satu atau lebih pertanyaan belum dijawab. Mohon periksa kembali.",
            )
            self._log_event(
                action_type="passive",
                event_type="validation_error",
                details={"message": "Survei belum lengkap (ada jawaban kosong)."},
            )
            return

        summary_paragraph = (
            "Terima kasih telah meluangkan waktu untuk check-in ini. "
            "Berikut adalah ringkasan dari apa yang telah Anda bagikan:\n\n"
        )

        for i, question_data in enumerate(self.questions):
            question_short = question_data["text"].split(". ", 1)[1] if ". " in question_data["text"] else question_data["text"]
            answer = self.user_answers[i]
            summary_paragraph += f"Mengenai '{question_short}', Anda menyampaikan bahwa '{answer}'. "

        summary_paragraph += (
            "\n\nPerlu diingat, ini adalah ruang untuk merefleksikan diri, dan setiap perasaan atau pikiran yang muncul adalah valid. "
            "Jika ada hal yang terasa membebani atau ingin Anda diskusikan lebih lanjut, jangan ragu untuk mencari dukungan dari profesional kesehatan mental. "
            "Langkah ini adalah investasi penting untuk kesejahteraan Anda."
            "\n\n<b>Pernyataan: Alat ini hanya untuk tujuan ilustrasi dan bukan pengganti nasihat, diagnosis, atau pengobatan medis profesional.</b>"
        )

        dialog = QMessageBox(self)
        dialog.setStyleSheet(self.styleSheet())
        dialog.setWindowTitle("Check-in Selesai")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.RichText)
        dialog.setText(summary_paragraph)
        dialog.exec()

        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self._log_event(action_type="passive", event_type="survey_completed")
        self.close()

    def closeEvent(self, event):
        self._stop_webcam_capture()
        self._log_event(
            action_type="passive", event_type="application_closed"
        )
        print(
            f"Aplikasi ditutup. Log survei: '{self.survey_log_file_name}', Log prediksi: '{self.prediction_log_file_name}', Log jawaban terbuka: '{self.answer_log_file_name}'."
        )
        event.accept()


# --- Function to Display Survey Log (Outside the App Class) ---
def display_survey_log(log_file_name):
    print(
        f"\n--- Mencoba Menampilkan Log Interaksi Survei dari '{log_file_name}' ---"
    )
    try:
        if not os.path.exists(log_file_name) or os.path.getsize(log_file_name) == 0:
            print(f"File log '{log_file_name}' tidak ditemukan atau kosong.")
            return
        with open(log_file_name, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if not isinstance(logs, list) or not logs:
            print(f"Tidak ada log yang valid ditemukan di file JSON: {log_file_name}.")
            return
        print(f"\nLog Interaksi dari '{log_file_name}':")
        print("-" * 80)
        print(f"{'Timestamp':<25} | {'Aksi':<8} | {'Tipe Event':<25} | {'Detail'}")
        print("-" * 80)
        for log_entry in logs:
            ts = log_entry.get("timestamp", "N/A")
            action = log_entry.get("action_type", "N/A")
            event_type_val = log_entry.get("event_type", "N/A")
            details_dict = log_entry.get("details", {})
            details_str_parts = []
            if isinstance(details_dict, dict):
                for key, value in details_dict.items():
                    details_str_parts.append(f"{key}: {value}")
            elif details_dict:
                details_str_parts.append(str(details_dict))
            details_str = ", ".join(details_str_parts) if details_str_parts else ""
            print(f"{ts:<25} | {action:<8} | {event_type_val:<25} | {details_str}")
        print("-" * 80)
    except json.JSONDecodeError:
        print(
            f"Error: Tidak dapat mendekode JSON dari '{log_file_name}'. File mungkin rusak."
        )
    except Exception as e:
        print(
            f"Terjadi kesalahan tak terduga saat menampilkan log dari '{log_file_name}': {e}"
        )


# --- Function to Display Prediction Log ---
def display_prediction_log(log_file_name):
    print(
        f"\n--- Mencoba Menampilkan Log Prediksi Gambar dari '{log_file_name}' ---"
    )
    try:
        if not os.path.exists(log_file_name) or os.path.getsize(log_file_name) == 0:
            print(f"File log prediksi '{log_file_name}' tidak ditemukan atau kosong.")
            return

        with open(log_file_name, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        if not isinstance(predictions, list) or not predictions:
            print(f"Tidak ada prediksi valid ditemukan di file JSON: {log_file_name}.")
            return

        print(f"\nLog Prediksi Gambar dari '{log_file_name}':")
        print("-" * 60)
        print(f"{'Timestamp':<25} | {'Label Prediksi':<20} | {'Keyakinan':<10}")
        print("-" * 60)
        for entry in predictions:
            ts = entry.get("timestamp", "N/A")
            label = entry.get("predicted_label", "N/A")
            confidence = entry.get("confidence", "N/A")
            print(f"{ts:<25} | {label:<20} | {confidence:<10.4f}")
        print("-" * 60)

    except json.JSONDecodeError:
        print(
            f"Error: Tidak dapat mendekode JSON dari log prediksi '{log_file_name}'. File mungkin rusak."
        )
    except Exception as e:
        print(
            f"Terjadi kesalahan tak terduga saat menampilkan log prediksi dari '{log_file_name}': {e}"
        )


# --- Function to Display Open Answers Log ---
def display_open_answers_log(log_file_name):
    print(
        f"\n--- Mencoba Menampilkan Log Jawaban Terbuka dari '{log_file_name}' ---"
    )
    try:
        if not os.path.exists(log_file_name) or os.path.getsize(log_file_name) == 0:
            print(f"File log jawaban terbuka '{log_file_name}' tidak ditemukan atau kosong.")
            return
        with open(log_file_name, "r", encoding="utf-8") as f:
            answers = json.load(f)
        if not isinstance(answers, list) or not answers:
            print(f"Tidak ada jawaban valid ditemukan di file JSON: {log_file_name}.")
            return

        print(f"\nLog Jawaban Terbuka dari '{log_file_name}':")
        print("-" * 90)
        print(f"{'Timestamp':<25} | {'Pertanyaan ke':<15} | {'Pertanyaan':<30} | {'Jawaban'}")
        print("-" * 90)
        for entry in answers:
            ts = entry.get("timestamp", "N/A")
            q_idx = entry.get("question_index", "N/A")
            question_text = entry.get("question", "N/A")
            answer_text = entry.get("answer", "N/A")

            q_display = question_text[:27] + "..." if len(question_text) > 30 else question_text
            a_display = answer_text[:30] + "..." if len(answer_text) > 30 else answer_text

            print(f"{ts:<25} | {str(q_idx):<15} | {q_display:<30} | {a_display}")
        print("-" * 90)
    except json.JSONDecodeError:
        print(
            f"Error: Tidak dapat mendekode JSON dari log jawaban terbuka '{log_file_name}'. File mungkin rusak."
        )
    except Exception as e:
        print(
            f"Terjadi kesalahan tak terduga saat menampilkan log jawaban terbuka dari '{log_file_name}': {e}"
        )


if __name__ == "__main__":
    run_application = True
    if run_application:
        app_instance = QApplication(sys.argv)
        survey_app = ModernMentalHealthSurveyApp()
        survey_app.show()
        exit_code = app_instance.exec()
        if exit_code == 0:
            print(f"\nAplikasi selesai.")
            display_survey_log(survey_app.survey_log_file_name)
            display_prediction_log(
                survey_app.prediction_log_file_name
            )
            display_open_answers_log(
                survey_app.answer_log_file_name
            )
        sys.exit(exit_code)
    else:
        print("Set run_application = True untuk menjalankan survei.")
        sys.exit(0)