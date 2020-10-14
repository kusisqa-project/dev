import DataLoader
from emoclass import EmoClassification as ec


if __name__ == "__main__":
    data = {'dataset': 'deap', 'classifier': 'svm', 'signals': ['GSR'], 'winSize': 63, 'winIni': 0, 'sampleSize': 128}
    dataset_folder = "datasets/deap_preprocessed/"
    data_folder = "datasets/data_files/"
    models_folder = "models/"
    loadDataset = False
    if loadDataset:
        DataLoader.convert_dataset(data["dataset"], dataset_folder, data_folder)

    features = ec.extractFeatures(data, data_folder)
    ec.train_and_test(data['classifier'], data_folder, models_folder)
