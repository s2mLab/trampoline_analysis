import os
import re

import csv
import numpy as np


class DataReader:
    @staticmethod
    def read_cycl_data(filepath) -> np.ndarray:
        out = []
        with open(filepath) as csvfile:
            rows = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                out.append([float(i) for i in row])
        return np.array(out)

    @staticmethod
    def read_gl_data(filepath) -> np.ndarray:
        out = []
        with open(filepath) as csvfile:
            rows = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                out.append([float(i) for i in row])
        return np.array(out)

    @staticmethod
    def fetch_trial_names(folder) -> tuple[str, ...]:
        all_filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        root_names = [re.split(r"^([0-9_.]*)_([a-zA-Z_.]*)(.CSV)$", filename)[1] for filename in all_filenames]
        return tuple(set(root_names))
