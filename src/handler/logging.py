import json
import os
from datetime import datetime
from typing import Literal, Optional, Dict, Any


class LoggingHandler:
    """
    A class to handle logging. It creates a log directory if it does not exist and generates unique filenames
    for each session based on the current timestamp.
    """

    def __init__(self, taskname: str, timestamp: Optional[str] = None):
        self.__log_directory = "logs"
        self.taskname = taskname
        self.__setup_log_directory()
        self.__generate_log_filenames(timestamp=timestamp)
        self.__logs = []

    def __setup_log_directory(self):
        try:
            os.makedirs(self.__log_directory, exist_ok=True)
            print(
                f"Logs will be saved in directory: '{os.path.abspath(self.__log_directory)}'"
            )
        except OSError as e:
            print(
                f"Error creating log directory '{self.__log_directory}': {e}. Logs will be saved in current directory."
            )
            self.__log_directory = "."

    def __generate_log_filenames(self, timestamp: Optional[str] = None):
        base_timestamp_str = timestamp
        if base_timestamp_str is None:
            base_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_name = os.path.join(
            self.__log_directory, f"{self.taskname}_log_{base_timestamp_str}.json"
        )
        print(f"Current session {self.taskname} log file: {self.log_file_name}")

    def write_log_event(
        self,
        action_type: Literal["active", "passive"],
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "event_type": event_type,
        }
        if details:
            log_entry["details"] = details

        self.__logs.append(log_entry)

    def save_log(self):
        try:
            with open(self.log_file_name, "w", encoding="utf-8") as f:
                json.dump(self.__logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing to log file '{self.log_file_name}': {e}")


class PHQLogging(LoggingHandler):
    def __init__(self, timestamp: Optional[str] = None):
        super().__init__(taskname="phq", timestamp=timestamp)
        self.__current_index = None

    def display_question(self, question_idx: Optional[int] = None):
        if self.__current_index is not None:
            self.write_log_event(
                action_type="passive",
                event_type="question_closed",
                details={"index": self.__current_index},
            )
            self.__current_index = question_idx
        if question_idx is not None:
            self.write_log_event(
                action_type="passive",
                event_type="question_opened",
                details={"index": question_idx},
            )

    def select_option(self, question_idx: int, option_text: str):
        self.write_log_event(
            action_type="active",
            event_type="option_selected",
            details={
                "question_index": question_idx,
                "selected_option": option_text,
            },
        )


class OpenQuestionLogging(LoggingHandler):
    def __init__(self, timestamp: Optional[str] = None):
        super().__init__(taskname="open_question", timestamp=timestamp)
        self.__current_index = None

    def display_question(self, question_idx: Optional[int] = None):
        if self.__current_index is not None:
            self.write_log_event(
                action_type="passive",
                event_type="question_closed",
                details={"index": self.__current_index},
            )
            self.__current_index = question_idx
        if question_idx is not None:
            self.write_log_event(
                action_type="passive",
                event_type="question_opened",
                details={"index": question_idx},
            )


class EmotionLogging(LoggingHandler):
    def __init__(self, timestamp: Optional[str] = None):
        super().__init__(taskname="emotion", timestamp=timestamp)

    def add_label(self, label: str, confidence: float):
        self.write_log_event(
            action_type="passive",
            event_type="emotion_detected",
            details={
                "label": label,
                "confidence": round(confidence, 4),
            },
        )
