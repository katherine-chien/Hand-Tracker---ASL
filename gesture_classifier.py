import numpy as np
import joblib

class GestureClassifier:

    def __init__(self, model_path="model.pkl"):
        try:
            self.model = joblib.load(model_path)
        except:
            self.model = None

    def predict(self, landmarks):

        if self.model is None:
            return "?"

        landmarks = np.array(landmarks).flatten().reshape(1, -1)

        prediction = self.model.predict(landmarks)

        return prediction[0]
