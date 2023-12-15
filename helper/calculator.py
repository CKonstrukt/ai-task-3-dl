import os
import pandas as pd
import numpy as np

class ImageCount:
    def __init__(self, folder):
        self.folder = folder

    def count(self):
        folders = os.listdir(self.folder)
        dictio = {}

        for subfolder in folders:
            subfolder_path = os.path.join(self.folder, subfolder)
            if os.path.isdir(subfolder_path):
                file_count = len(os.listdir(subfolder_path))
                dictio[subfolder] = file_count

        df = pd.DataFrame.from_dict(dictio, orient='index', columns=['File Count'])
        return df
