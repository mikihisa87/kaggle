import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

path = '/Users/user/python/kaggle/TalkingData/input/'
dtypes = {
    'ip'    : 'uint32',
    'app'   : 'uint16',
    'device': 'uint16',
    'os'    : 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

print('load train....')
train_df = pd.read_csv(path+'train.csv', dtype=dtypes, skiprows=range(1, 131886954), usecols=
['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
print('load test....')
test_df = pd.read_csv(path+'test.csv', dtype=dtypes, usecols=
['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
len_train = len(train_df)
train_df = train_df.append(test_df)
del test_df; gc.collect()

print('hour, day, wday....')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['wday'] = pd.to_datetime(train_df.click_time).dt.dayofweek.astype('uint8')

print('grouping by ip-day-hour combination....')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp; gc.collect()

print('group by ip-app combination....')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip', 'app'], how='left')
del gp; gc.collect()

print('group by ip-app-os combination....')
gp = train_df[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip', 'app', 'os'], how='left')
del gp; gc.collect()

print('vars and data type....')
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

print('label encoding....')
from sklearn.preprocessing import LabelEncoder
train_df[['app', 'device', 'os', 'channel', 'hour', 'day', 'wday']].apply(LabelEncoder().fit_transform)

print('final part of preparation....')
test_df = train_df[len_train:]
train_df = train_df[:len_train]
y_train = train_df['is_attributed'].values
train_df.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)

print('neural network....')
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['channel'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_wd = np.max([train_df['wday'].max(), test_df['wday'].max()])+1
max_qty = np.max([train_df['qty'].max(), test_df['qty'].max()])+1
max_c1 = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_c2 = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name='app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name='ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name='dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name='os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name='h')
emb_h = Embedding(max_h, emb_n)(in_h)
in_d = Input(shape=[1], name='d')
emb_d = Embedding(max_d, emb_n)(in_d)
in_wd = Input(shape=[1], name='wd')
emb_wd = Embedding(max_wd, emb_n)(in_wd)
in_qty = Input(shape=[1], name='qty')
emb_qty = Embedding(max_qty, emb_n)(in_qty)
in_c1 = Input(shape=[1], name='c1')
emb_c1 = Embedding(max_c1, emb_n)(in_c1)
in_c2 = Input(shape=[1], name='c2')
emb_c2 = Embedding(max_c2, emb_n)(in_c2)

fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h),
                (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
#s_dout = SpatialDropout1D(0.2)(fe)

#----CapsNet Setup-----------------
from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, u_hat_vecs, [2, 3])
                #清水さん修正パート

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

# x = Sequential(inp.prm1, activation='relu', dropout=inp.prm5, recurrent_dropout=inp.prm5,
# return_sequences=True)(embed_layer)

capsule = Capsule(num_capsule=10, dim_capsule=16, routings=5, share_weights=True)(fe)
capsule = Flatten()(capsule)
capsule = Dropout(0.25)(capsule)
# x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
# x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
outp = Dense(1, activation='sigmoid')(capsule)
model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h, in_d, in_wd, in_qty, in_c1, in_c2], outputs=outp)

batch_size = 20000
epochs = 4
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1))-1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizers_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy', optimizer=optimizers_adam, metrics=['accuracy'])

model.summary()

model.fit(train_df, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
del train_df, y_train; gc.collect()
model.save_weights('dl_support.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id', 'click_time', 'ip', 'is_attributed'], 1, inplace=True)
test_df = get_keras_data(test_df)

print('predicting....')
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print('writing....')
sub.to_csv('20180414_dnn_capsnet_model5.csv', index=False)
