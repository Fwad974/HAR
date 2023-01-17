import os
import shutil
from scipy import io
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
from sklearn.svm import SVC



def data_loader(dataset_name="adl",
                verbose=True,
                incl_xyz_accel=True,  # include component accel_x/y/z in ____X data
                incl_val_group=False,  # True => returns x/y_test, x/y_validation, x/y_train
                # False => combine test & validation groups
                split_subj=dict
                    (train_subj=[4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 19, 20, 21, 22, 24, 26, 27, 29],
                     validation_subj=[1, 9, 16, 23, 25, 28],
                     test_subj=[2, 3, 13, 17, 18, 30]),
                normalize=False  # True returns x,y,z/norm(x,y,z)
                ):
    if (not os.path.isdir('./dataset/UniMiB-SHAR')):
        shutil.unpack_archive('UniMiB-SHAR.zip', '.', 'zip')
    # Convert .mat files to numpy ndarrays
    path_in = 'dataset/UniMiB-SHAR/data'
    # loadmat loads matlab files as dictionary, keys: header, version, globals, data
    adl_data = io.loadmat(path_in + '/' + dataset_name + '_data.mat')[dataset_name + '_data']
    adl_names = io.loadmat(path_in + '/' + dataset_name + '_names.mat', chars_as_strings=True)[dataset_name + '_names']
    adl_labels = io.loadmat(path_in + '/' + dataset_name + '_labels.mat')[dataset_name + '_labels']

    if (verbose):
        headers = ("Raw data", "shape", "object type", "data type")
        mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                  ("adl_labels:", adl_labels.shape, type(adl_labels), adl_labels.dtype),
                  ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
        print(tabulate(mydata, headers=headers))
    # Reshape data and compute total (rms) acceleration
    num_samples = 151
    # UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
    adl_data = np.reshape(adl_data, (-1, num_samples, 3), order='F')  # uses Fortran order

    if normalize:
        rms_accel = np.sqrt((adl_data[:, :, 0] ** 2) + (adl_data[:, :, 1] ** 2) + (adl_data[:, :, 2] ** 2))
        for i in range(3):
            adl_data[:, :, i] = adl_data[:, :, i] / rms_accel
    if (not incl_xyz_accel):
        adl_data = np.delete(adl_data, [0, 1, 2], 2)
    print("ADL : ", adl_data.shape)
    if (verbose):
        headers = ("Reshaped data", "shape", "object type", "data type")
        mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                  ("adl_labels:", adl_labels.shape, type(adl_labels), adl_labels.dtype),
                  ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
        print(tabulate(mydata, headers=headers))
    # Split train/test sets, combine or make separate validation set
    # ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
    # https://numpy.org/doc/stable/reference/generated/numpy.isin.html

    act_num = (adl_labels[:, 0]) - 1  # matlab source was 1 indexed, change to 0 indexed
    sub_num = (adl_labels[:, 1])

    test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
    x_test = adl_data[test_index]
    y_test = act_num[test_index]

    if (not incl_val_group):
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] +
                                         split_subj['validation_subj']))
        x_train = adl_data[train_index, :, 0]
        y_train = act_num[train_index]

        res = {"train": {"x": x_train, "y": y_train},
               "test": {"x": x_test, "y": y_test}
               }

    else:
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]

        validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
        x_validation = adl_data[validation_index]
        y_validation = act_num[validation_index]
        res = {"train": {"x": x_train, "y": y_train},
               "valid": {"x": x_validation, "y": y_validation},
               "test": {"x": x_test, "y": y_test}
               }

    return res


def white(data):
    for z, i in enumerate(data):
        data[z] = whiten(i)
    return data

if __name__=="main":

    

    data = data_loader()

    x_train = data["train"]["x"]
    x_train = white(x_train)

    x_test = data["test"]["x"]
    x_test = white(x_test)

    features  = x_train.reshape(-1,3)
    cook,diff=kmeans(features,1024,seed=0)


    code_data = {"train": [], "test": []}
    raw_data = {"train": x_train, "test": x_test}
    for typ in code_data:
        for i in raw_data[typ]:
            code, _ = vq(i, cook)
            code_data[typ].append(code)


    lda = LatentDirichletAllocation(n_components=200,
                                    random_state=0)
    lda.fit(code_data["train"])

    lda_data = {}
    for typ in code_data:
        lda_data[typ] = lda.transform(code_data[typ])



    X = lda_data["train"]
    y = data["train"]["y"]


    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)

    labels = clf.predict(lda_data["test"])
    accuracy_score(labels, data["test"]["y"])
