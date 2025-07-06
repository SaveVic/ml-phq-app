def apply_styles(widget):
    qss = """
        QWidget#MainWindow { background-color: #F4F6F8; font-family: 'Segoe UI', Arial, sans-serif; }
        QLabel#DisclaimerLabel { background-color: #FFF3CD; color: #664D03; border: 1px solid #FFECB5; border-radius: 6px; padding: 12px; font-size: 9pt; font-weight: normal; }
        QLabel#InstructionsLabel { color: #4A5568; font-size: 11pt; padding-left: 5px; margin-bottom: 0px; }
        QWidget#QuestionAreaWidget { background-color: #FFFFFF; border-radius: 8px; border: 1px solid #E2E8F0; }
        QLabel#QuestionLabel { color: #1A202C; font-size: 13pt; font-weight: bold; line-height: 1.5; padding-bottom: 10px; }
        QLabel#ProgressLabel { color: #718096; font-size: 9pt; font-weight: bold; }
        QRadioButton { font-size: 11pt; color: #2D3748; padding: 8px 5px; spacing: 10px; }
        QRadioButton::indicator { width: 18px; height: 18px; }
        QRadioButton::indicator:unchecked { border: 2px solid #A0AEC0; border-radius: 9px; background-color: #FFFFFF; }
        QRadioButton::indicator:unchecked:hover { border: 2px solid #718096; }
        QRadioButton::indicator:checked { border: 2px solid #3182CE; border-radius: 9px; background-color: #3182CE; }
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
    widget.setStyleSheet(qss)
