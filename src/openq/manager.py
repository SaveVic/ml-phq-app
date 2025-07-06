from copy import deepcopy
from .questions import OPEN_QUESTIONS_LIST


class OpenQuestionManager:
    def __init__(self):
        pass

    def get_questions(self):
        return deepcopy(OPEN_QUESTIONS_LIST)
