import tensorflow.keras as keras
import numpy as np
import librosa

### For no warning prints of librosa.feature.mfcc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "one",
        "four",
        "go",
        "off",
        "right",
        "six",
        "five",
        "down",
        "cat",
        "three",
        "no",
        "left",
        "two",
        "on",
        "zero",
        "nine",
        "dog",
        "stop",
        "seven",
        "up",
        "eight",
        "yes"
    ]

    _instance = None

    def preprocess(self, file_path, samples_to_consider = 22050, n_mfcc = 13, n_fft = 2048, hop_length = 512):
        
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > samples_to_consider:
            signal = signal[:samples_to_consider]        

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, hop_length = hop_length, n_fft = n_fft)

        return MFCCs.T

    def predict(self, file_path):
        
        # extract MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # MFCCs)

        # convert 2D MFCC array to 4D array -> (# samples, # segments, # MFCCs, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction 
        predictions = self.model.predict(MFCCs) 
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

def Keyword_Spotting_Service(model_path = 'CNNmodel.h5'):

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(model_path)
    
    return _Keyword_Spotting_Service._instance

# ---

if __name__ == '__main__':

    model_path = 'CNNmodel.h5'
    kss = Keyword_Spotting_Service(model_path)

    prediction_down = kss.predict('test_prediction/down.wav')
    print(f'Audio: down / Prediction: {prediction_down}')

    prediction_nine = kss.predict('test_prediction/nine.wav')
    print(f'Audio: nine / Prediction: {prediction_nine}')

    