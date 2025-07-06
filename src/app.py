from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QMessageBox,
    QSpacerItem,
    QApplication,
    QSizePolicy,
)
from PyQt6.QtCore import Qt

from consts import WINDOW_TITLE
from .handler.model import ModelHandler
from .handler.webcam import WebcamHandler
from .handler.logging import SurveyLogging
from .ui import apply_styles
from .phq.manager import PHQ_QUESTIONS


class ModernMentalHealthSurveyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)

        # Initialize handlers
        self.survey_logging = SurveyLogging()
        self.model_handler = ModelHandler()
        self.webcam_handler = WebcamHandler(self.model_handler)

        # Setup UI
        self.questions = PHQ_QUESTIONS
        self.num_questions = len(self.questions)
        self.current_question_index = 0
        self.user_answers = [None] * self.num_questions

        self.init_ui()
        apply_styles(self)
        self._center_window()

        # Start webcam
        self.webcam_handler.start_capture()

        self.display_question()

    def _center_window(self):
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            center_point = screen_geometry.center()
            self.move(
                center_point.x() - self.width() // 2,
                center_point.y() - self.height() // 2,
            )

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # Disclaimer label
        self.disclaimer_label = QLabel(
            "<b>Important:</b> This is NOT a diagnostic tool. Consult a professional for any mental health concerns."
        )
        self.disclaimer_label.setObjectName("DisclaimerLabel")
        self.disclaimer_label.setWordWrap(True)
        self.main_layout.addWidget(self.disclaimer_label)

        # Instructions label
        self.instructions_label = QLabel(
            "Please answer the following questions based on how you have felt over the <b>last 2 weeks</b>."
        )
        self.instructions_label.setObjectName("InstructionsLabel")
        self.instructions_label.setWordWrap(True)
        self.main_layout.addWidget(self.instructions_label)

        # Question area
        self.question_area_widget = QWidget()
        self.question_area_widget.setObjectName("QuestionAreaWidget")
        question_area_layout = QVBoxLayout(self.question_area_widget)
        question_area_layout.setContentsMargins(20, 15, 20, 15)
        question_area_layout.setSpacing(15)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("ProgressLabel")
        question_area_layout.addWidget(
            self.progress_label, alignment=Qt.AlignmentFlag.AlignRight
        )

        # Question label
        self.question_label = QLabel("Question text will appear here.")
        self.question_label.setObjectName("QuestionLabel")
        self.question_label.setWordWrap(True)
        question_area_layout.addWidget(self.question_label)

        # Options layout
        self.options_layout_container = QVBoxLayout()
        self.options_layout_container.setSpacing(10)
        self.radio_button_group = QButtonGroup(self)
        self.radio_button_group.setExclusive(True)
        question_area_layout.addLayout(self.options_layout_container)

        # Spacer
        question_area_layout.addSpacerItem(
            QSpacerItem(5, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        self.main_layout.addWidget(self.question_area_widget)

        # Navigation buttons
        self.nav_buttons_layout = QHBoxLayout()
        self.nav_buttons_layout.setSpacing(10)
        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("PrevButton")
        self.prev_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_button.clicked.connect(self.go_previous)
        self.nav_buttons_layout.addSpacerItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        self.nav_buttons_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("NextButton")
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.go_next)
        self.nav_buttons_layout.addWidget(self.next_button)
        self.main_layout.addLayout(self.nav_buttons_layout)

        self.setMinimumSize(550, 500)
        self.resize(600, 550)

    def display_question(self):
        # Clear existing options
        for button in self.radio_button_group.buttons():
            self.radio_button_group.removeButton(button)
            button.deleteLater()
        while self.options_layout_container.count():
            child = self.options_layout_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.current_question_index < self.num_questions:
            question_data = self.questions[self.current_question_index]
            self.question_label.setText(question_data["text"])
            self.progress_label.setText(
                f"Question {self.current_question_index + 1} of {self.num_questions}"
            )

            # Add radio buttons for options
            for option_text in question_data["options"]:
                radio_button = QRadioButton(option_text)
                radio_button.setCursor(Qt.CursorShape.PointingHandCursor)
                self.options_layout_container.addWidget(radio_button)
                self.radio_button_group.addButton(radio_button)
                radio_button.toggled.connect(
                    lambda checked, q_idx=self.current_question_index, opt_txt=option_text: self._handle_option_toggled(
                        checked, q_idx, opt_txt
                    )
                )
                if self.user_answers[self.current_question_index] == option_text:
                    radio_button.setChecked(True)

            # Update button states
            self.prev_button.setEnabled(self.current_question_index > 0)
            if self.current_question_index == self.num_questions - 1:
                self.next_button.setText("Finish")
            else:
                self.next_button.setText("Next")

            # Log question display
            self.survey_logging.display_question(self.current_question_index)
        else:
            self.process_survey()

    def _handle_option_toggled(self, checked, question_idx, option_text):
        if checked:
            self.user_answers[question_idx] = option_text
            self.survey_logging.select_option(question_idx, option_text)

    def go_next(self):
        if self.user_answers[self.current_question_index] is None:
            QMessageBox.warning(
                self, "No Answer", "Please select an answer before proceeding."
            )
            return

        if self.current_question_index < self.num_questions - 1:
            self.current_question_index += 1
            self.display_question()
        else:
            self.process_survey()

    def go_previous(self):
        if self.current_question_index > 0:
            self.current_question_index -= 1
            self.display_question()

    def process_survey(self):
        self.survey_logging.display_question(None)

        if any(answer is None for answer in self.user_answers):
            QMessageBox.warning(
                self,
                "Incomplete Survey",
                "One or more questions were not answered. Please review.",
            )
            return

        yes_count = 0
        responses_summary = []
        for i, answer in enumerate(self.user_answers):
            question_text_full = self.questions[i]["text"]
            question_text_short = (
                question_text_full.split(". ", 1)[1]
                if ". " in question_text_full
                else question_text_full
            )
            responses_summary.append(f"- {question_text_short}: {answer}")
            if answer == "Yes":
                yes_count += 1

        result_message_intro = (
            "Thank you for completing the check-in.\n\nYour responses:\n"
            + "\n".join(responses_summary)
            + "\n\n"
        )

        if yes_count >= 2:
            result_message_intro += (
                "Based on your responses, it might be helpful to talk to someone about how you're feeling. "
                + "Remember, support is available, and speaking with a qualified healthcare professional can provide guidance."
            )
        else:
            result_message_intro += (
                "Thank you for taking the time for this check-in. Remember to prioritize your well-being. "
                + "If you ever feel overwhelmed or have concerns, reaching out to a healthcare professional is a positive step."
            )

        result_message_intro += "\n\n<b>Disclaimer: This tool is for illustrative purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.</b>"

        dialog = QMessageBox(self)
        dialog.setStyleSheet(self.styleSheet())
        dialog.setWindowTitle("Check-in Complete")
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.RichText)
        dialog.setText(result_message_intro)
        dialog.exec()

        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)

        self.logging_handler.log_event(
            action_type="passive", event_type="survey_completed", survey_log=True
        )
        self.close()

    def closeEvent(self, event):
        self.webcam_handler.stop_capture()
        print(
            f"Application closed. Survey log: '{self.survey_logging.log_file_name}', "
            # f"Prediction log: '{self.logging_handler.prediction_log_file_name}'."
        )
        event.accept()
