import cv2
import numpy as np 
import os

class SimpleDatasetLoader:
    def __init__(self , preprocessors = None):
        self.preprocessors = preprocessors

        if preprocessors is None:
            self.preprocessors = []

    def load(self , imagePaths, verbose = -1):
        data = []
        labels = []

        for i , imagePath in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None :
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if i > 0 and verbose > 1 and (i + 1)% verbose == 0:
                print('[INFO] processed {} / {}'.format(i + 1, len(imagePaths)))
            
        return (np.array(data), np.array(labels))
