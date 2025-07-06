from copy import deepcopy
import random
import pandas as pd
from .options import PHQ_OPTIONS


class PHQManager:
    def __init__(self, source_path: str | None = None):
        self.__source_path = source_path if source_path else "./questions.csv"
        self.__load_df()

    def __load_df(self):
        df = pd.read_csv(self.__source_path)
        indexes = df["id"].unique()
        cols = ["text", "aug_text"]
        self.__questions = {idx: df[df["id"] == idx][cols].values for idx in indexes}

    def get_questions(self):
        selected_questions = {
            idx: random.choice(self.__questions[idx]) for idx in self.__questions.keys()
        }
        return selected_questions

    def get_options(self):
        return deepcopy(PHQ_OPTIONS)
