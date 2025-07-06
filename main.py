import sys
from PyQt6.QtWidgets import QApplication
from .src.app import ModernMentalHealthSurveyApp
from .src.handler.logging import display_survey_log, display_prediction_log

if __name__ == "__main__":
    run_application = True
    if run_application:
        app_instance = QApplication(sys.argv)
        survey_app = ModernMentalHealthSurveyApp()
        survey_app.show()
        exit_code = app_instance.exec()
        if exit_code == 0:
            print(f"\nApplication finished.")
            display_survey_log(survey_app.survey_log_file_name)
            display_prediction_log(survey_app.prediction_log_file_name)
        sys.exit(exit_code)
    else:
        print("Set run_application = True to run the survey.")
        sys.exit(0)
