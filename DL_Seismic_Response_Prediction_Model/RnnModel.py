
from keras.models import Model
from keras.layers import Dense, LSTM, Conv1D, GRU, Bidirectional, Activation, Dropout, BatchNormalization
from keras.layers import Input, Masking, TimeDistributed, concatenate, Flatten, Reshape


class RnnModel(object):
    def __init__(self, num_gm_layer, num_concat_layer, num_fc_layer, gm_shape,br_shape, gm_layer_rnn, concat_layer_rnn, layer_fc, drop_rate):
        self.drop_rate = drop_rate
        self.num_gm_layer = num_gm_layer
        self.num_concat_layer = num_concat_layer
        self.num_fc_layer = num_fc_layer
        self.gm_shape = gm_shape
        self.br_shape = br_shape
        self.gm_layer_rnn = gm_layer_rnn
        self.concat_layer_rnn = concat_layer_rnn
        self.layer_fc = layer_fc

    def createModel(self):

        GM_features = Input(shape=self.gm_shape)
        BR_input = Input(shape=self.br_shape)
        for i in range(self.num_gm_layer):
            GM_features = GRU(units=self.gm_layer_rnn[i], return_sequences=True)(GM_features)  # GRU (use 128 units and return the sequences)
            GM_features = BatchNormalization()(GM_features)  # Batch normalization
            GM_features = Dropout(rate=self.drop_rate)(GM_features)  # dropout (use 0.8)

        concatenated = concatenate([GM_features, BR_input], axis=2)
        for i in range(self.num_concat_layer):
            concatenated = GRU(units=self.concat_layer_rnn[i], return_sequences=True)(concatenated)
            concatenated = BatchNormalization()(concatenated)  # Batch normalization
            concatenated = Dropout(rate=self.drop_rate)(concatenated)

        fc = Flatten()(concatenated)
        for i in range(self.num_fc_layer):
            fc = Dense(self.layer_fc[i], activation='relu')(fc)

        output = Dense(1)(fc)
        model = Model(inputs=[GM_features, BR_input], outputs=[output])
        return model

