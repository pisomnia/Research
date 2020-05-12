from getConfig import get_config
from data_utils import generate_data
from RnnModel import RnnModel
from plot_utils import score, plot_result

import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.models import load_model
import matplotlib.pyplot as plt

gConfig = {}
gConfig = get_config(config_file='config.ini')


def create_model():
    if 'pretrained_model' in gConfig:
        model = load_model(gConfig['pretrained_model'])
        return model
    ckpt = os.listdir(gConfig['working_directory'])

    if len(ckpt) != 0:
        model_file = os.path.join(gConfig['working_directory'], ckpt[-1])
        print("Reading model parameters from %s" % model_file)
        model = tf.keras.models.load_model(model_file)
        return model
    else:
        rnn_model = RnnModel(gConfig['num_gm_layer'], gConfig['num_concat_layer'], gConfig['num_fc_layer'], gConfig['gm_shape'], gConfig['br_shape'], gConfig['gm_layer_rnn'], gConfig['concat_layer_rnn'], gConfig['layer_fc'], gConfig['drop_rate'])
        model = rnn_model.createModel()
        return model

def train():
    model = create_model()
    print(model.summary())
    BR_train, GM_train, y_train, BR_test, GM_test, y_test = generate_data(gConfig['dataset_path'], gConfig['num_gm'], gConfig['gm_shape'])
    if gConfig['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if gConfig['optimizer'] == 'adam':
        opt = Adam(lr=gConfig['learning_rate'], decay=gConfig['lr_decay_momentum'])
    elif gConfig['optimizer'] == 'sgd':
        opt = SGD(lr=gConfig['learning_rate'], decay=gConfig['lr_decay_momentum'])
    else:
        opt = SGD(lr=0.005)

    model.compile(loss=gConfig['loss_function'], optimizer=opt)
    filepath = "rnn_model_{epoch:03d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(gConfig['working_directory'], filepath), monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)

    history = model.fit([GM_train, BR_train], y_train,
              batch_size=gConfig['batch_size'],
              epochs=gConfig['num_epochs'],
              verbose=2,
              validation_data=([GM_test, BR_test], y_test),
              callbacks=[checkpoint],
              shuffle=True)

    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    # filename = 'rnn_model.h5'
    # checkpoint_path = os.path.join(gConfig['working_directory'], filename)
    # model.save(checkpoint_path)

    ckpt = os.listdir(gConfig['working_directory'])
    model_file = os.path.join(gConfig['working_directory'], ckpt[-1])
    print("Best Model File: %s" % model_file)
    best_model = load_model(model_file)

    y_nn_test = best_model.predict([GM_test, BR_test])
    y_nn_test = y_nn_test.flatten()
    model_name = 'RNN Based Seismic Response Prediction Model'
    ARD = score(model_name, y_test, y_nn_test)


def predict(data):
    ckpt = os.listdir(gConfig['working_directory'])
    checkpoint_path = os.path.join(gConfig['working_directory'], ckpt[-1])
    model = load_model(checkpoint_path)
    res = model.predict(data)
    print('test after load: ', res)


# if __name__ == '__main__':
#     rnn_model = RnnModel(gConfig['num_gm_layer'], gConfig['num_concat_layer'], gConfig['num_fc_layer'],
#                          gConfig['gm_shape'], gConfig['br_shape'], gConfig['gm_layer_rnn'], gConfig['concat_layer_rnn'],
#                          gConfig['layer_fc'], gConfig['drop_rate'])
#     Model = rnn_model.createModel()
#     Model = create_model()

if __name__ == '__main__':
    if gConfig['mode'] == 'train':
        print("Start Training the Model")
        print("========================")
        train()
    elif gConfig['mode'] == 'predict':
        predict()



