import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def generate_data(path, num_GM, gm_shape):
    file_BR = 'Result_NR_300.csv'
    NR_data = pd.read_csv(file_BR, header=None)
    NR_data.columns = ['Weight', 'Height', 'K1', 'Q', 'K2', 'T', 'Q/W', 'Drift']
    NR_data.index = range(len(NR_data))
    print("Bridge Response Data Loaded")
    print(NR_data.head())

    # path_GM = path + "GM_Spectrogram/"   待读取的文件夹
    path_list = os.listdir(path)
    path_list.sort(key=lambda xx: int(xx[3:-4]))
    GM = np.zeros((num_GM, gm_shape[0], gm_shape[1]))

    for i in range(num_GM):
        x = loadmat(os.path.join(path, path_list[i]))
        GM[i] = x['m'].T

    NR_data = NR_data.drop(['Weight', 'Height', 'K1', 'Q', 'K2'], axis=1)
    idx = NR_data.index.values
    GM_data = GM[[idx % num_GM]]
    assert NR_data.shape[0] == GM_data.shape[0]

    NR_train, NR_test, GM_train, GM_test = train_test_split(NR_data, GM_data, test_size=0.2, shuffle=True)
    Summary = pd.DataFrame(np.array([[len(NR_train), len(NR_test)]]), columns=['Train', 'Test'], index=['Num'])
    print(Summary)

    BR_train = NR_train.iloc[:, :2].as_matrix()
    y_train = NR_train.iloc[:, 2].tolist()
    BR_test = NR_test.iloc[:, :2].as_matrix()
    y_test = NR_test.iloc[:, 2].tolist()

    BR_train = np.expand_dims(BR_train, axis=1)
    BR_test = np.expand_dims(BR_test, axis=1)
    BR_train = np.repeat(BR_train, 14, axis=1)
    BR_test = np.repeat(BR_test, 14, axis=1)
    return BR_train, GM_train, y_train, BR_test, GM_test, y_test

# if __name__=='__main__':
#     BR_train, GM_train, y_train, BR_test, GM_test, y_test = generate_data('dataset/', 300, (14, 36))
#     print(BR_train.shape)
#     print(GM_train.shape)
