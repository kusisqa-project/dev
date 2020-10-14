import pandas as pd
import numpy as np
import os
import re
import pickle
from sklearn.model_selection import train_test_split
import emoclass.ClassifiersManager as clfman
import features.GSR_feature_extract as gsr_fex

isOnline = True

featureExtractor = {
    'GSR': gsr_fex.extract_features
}


def processSignal(data):
    return 0


def extractFeatures(dataIn, data_folder):
    selectedSignals = dataIn["signals"]
    winSize = dataIn["winSize"]
    winIni = dataIn["winIni"]
    sampleSize = dataIn["sampleSize"]
    if isOnline:
        for signal in selectedSignals:
            if signal in featureExtractor.keys():
                pattern = re.compile("^" + signal)
                file_names = [f for f in os.listdir(data_folder) if pattern.match(f)]
                featureExtractor[signal](data_folder, file_names, winIni, winSize, sampleSize)

    features_df = pd.DataFrame()
    pattern = re.compile("^feat_")
    file_names = [f for f in os.listdir(data_folder) if pattern.match(f)]
    for fname in file_names:
        temp_features_df = pickle.load(open(data_folder + fname, 'rb'))
        features_df = pd.concat([features_df, temp_features_df], axis=1)

    # remove complex
    df_abs = features_df.select_dtypes(["complex128"]).apply(np.abs)
    list_drop = df_abs.columns
    features_df.drop(labels=list_drop, axis=1, inplace=True)
    features_df = pd.concat([df_abs, features_df], axis=1)

    features_df = features_df.fillna(0)
    pickle.dump(features_df, open(data_folder + "all_features_x", "wb"))
    return features_df


def selectFeatures(data):
    return 0


def classify(data, clf_valence, clf_arousal):
    valence = clf_valence.predict(data)
    arousal = clf_arousal.predict(data)
    return [{'valence': valence[i], 'arousal': arousal[i]} for i in range(len(valence))]


def train_and_test(idClf, data_folder, models_folder):
    y = pickle.load(open(data_folder + '/all_df_y', 'rb'))
    X = pickle.load(open(data_folder + '/all_features_x', 'rb'))
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    clfman.train_classifier(idClf, X_train.values.tolist(), y_train['valence'].values.tolist(), folder=models_folder, nameClf=idClf + '_valence')
    clfman.train_classifier(idClf, X_train.values.tolist(), y_train['arousal'].values.tolist(), folder=models_folder, nameClf=idClf + '_arousal')
    print("**** Valence classifier ****")
    clf = clfman.load_classifier(models_folder, idClf + '_valence')
    clfman.test_classifier(clf, X_test, y_test['valence'])
    print("**** Arousal classifier ****")
    clf = clfman.load_classifier(models_folder, idClf + '_arousal')
    clfman.test_classifier(clf, X_test, y_test['arousal'])


def initProcess(dataIn, models_folder, data_folder):
    # extract features
    features = extractFeatures(dataIn, data_folder)

    # load models
    clf_valence = clfman.load_classifier(models_folder, dataIn['classifier'] + '_valence')
    clf_arousal = clfman.load_classifier(models_folder, dataIn['classifier'] + '_arousal')

    # classify
    classes = classify(features, clf_valence, clf_arousal)
    feature_names = features.columns.tolist()

    return features, classes, feature_names


if __name__ == "__main__":
    data = {'signals': ['GSR'], 'winSize': 63, 'winIni': 0, 'sampleSize': 128}
    extractFeatures(data, '../datasets/data_files/')