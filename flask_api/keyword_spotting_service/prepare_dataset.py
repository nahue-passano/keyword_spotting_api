import librosa
import os
import json

### For no warning prints of librosa.feature.mfcc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_dataset(dataset_path, json_path, samples_to_consider, n_mfcc = 13, hop_length = 512, n_fft = 2048):
    """Generates a json file with the mfcc of each audio in dataset_path directory.

    Args:
        dataset_path (str): path of the dataset
        json_path (str): path of the json to be created
        n_mfcc (int, optional): Number of MFCC to be calculated. Defaults to 13.
        hop_length (int, optional): Hop length in number of samples. Defaults to 512.
        n_fft (int, optional): Number of samples to process de fft. Defaults to 2048.
    """

    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # we need to ensure that we're not at root level
        if dirpath is not dataset_path:

            # update mappings
            category = dirpath.split('/')[-1]
            data['mappings'].append(category)
            print(f'Processing {category}')

            # loop through all the filenames and extract MFCC's
            for f in filenames:

                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) > samples_to_consider:

                    # enforce 1 sec long signal
                    signal = signal[:samples_to_consider]

                    """ Why enforce 1 sec long?
                    Later we will use this dataset to train an CNN, so the dimensions
                    of the images (in this case de MFCC matrices) has to be all the
                    same dimension.
                    """

                    # extract the MFCC
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, hop_length = hop_length, n_fft = n_fft)

                    # store data
                    data['labels'].append(i-1)
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data['files'].append(file_path)
                    print(f'{file_path}: {i-1}')
    
    # store in json file
    with open(json_path,'w') as fp:
        json.dump(data, fp, indent = 4)


if __name__ == '__main__':

    ### Path and previous configuration
    dataset_path = 'flask_api/keyword_spotting_service/dataset_speech_recognition'
    json_path = 'flask_api/keyword_spotting_service/data.json'
    samples_to_consider = 22050 # 1 sec worth of sound

    prepare_dataset(dataset_path, json_path, samples_to_consider)