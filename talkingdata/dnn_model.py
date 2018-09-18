import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

nrows=184903891-1
nchunk=120000000
val_size=2500000

frm=nrows-120000000

to=frm+nchunk
predictors=[]
def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")

    GROUP_BY_NEXT_CLICKS = [

    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},

    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)

        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")

    GROUP_BY_NEXT_CLICKS = [

    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},

    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)

        predictors.append(new_feature)
        gc.collect()
    return (df)


## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract cumulative count feature  from different cols
def do_cumcount( df, group_cols, counted,agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted))
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted))
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


#path = '/Users/wolheelee/python/kaggle/TalkingData/input/'


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

print('loading train data...',frm,to)
train_df = pd.read_csv("/Users/wolheelee/python/kaggle/TalkingData/input/train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv("/Users/wolheelee/python/kaggle/TalkingData/input/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df=train_df.append(test_df)

del test_df

gc.collect()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8')
train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
train_df = do_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage.

train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_count( train_df, ['ip', 'day', 'hour'] ); gc.collect()
train_df = do_count( train_df, ['ip', 'app']); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os']); gc.collect()
# train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour'); gc.collect()
gp = train_df[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_cnt_channel'})
train_df = train_df.merge(gp, on=['ip'], how='left')
agg_name= 'ip_cnt_channel'
predictors.append(agg_name)
train_df['ip_app_device_os_channel_nextClick'] = train_df['ip_app_device_os_channel_nextClick'].fillna(0)
train_df['ip_os_device_nextClick'] = train_df['ip_os_device_nextClick'].fillna(0)
train_df['ip_os_device_app_nextClick'] = train_df['ip_os_device_app_nextClick'].fillna(0)
train_df['ip_channel_prevClick'] = train_df['ip_channel_prevClick'].fillna(0)
train_df['ip_os_prevClick'] = train_df['ip_os_prevClick'].fillna(0)
train_df['ip_app_device_os_channel_nextClick'] = train_df['ip_app_device_os_channel_nextClick'].astype(int)
train_df['ip_os_device_nextClick'] = train_df['ip_os_device_nextClick'].astype(int)
train_df['ip_os_device_app_nextClick'] = train_df['ip_os_device_app_nextClick'].astype(int)
train_df['ip_channel_prevClick'] = train_df['ip_channel_prevClick'].astype(int)
train_df['ip_os_prevClick'] = train_df['ip_os_prevClick'].astype(int)
del gp, agg_name
gc.collect()

print('label encoding....')
from sklearn.preprocessing import LabelEncoder
train_df[['app', 'device', 'os', 'channel', 'hour', 'day']].apply(LabelEncoder().fit_transform)

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

max_a_c_cntuni = np.max([train_df['app_by_channel_countuniq'].max(), test_df['app_by_channel_countuniq'].max()])+1
max_i_a_os_cntuni = np.max([train_df['ip_app_by_os_countuniq'].max(), test_df['ip_app_by_os_countuniq'].max()])+1
max_i_a_d_os_c_nxtcli = np.max([train_df['ip_app_device_os_channel_nextClick'].max(), test_df['ip_app_device_os_channel_nextClick'].max()])+1
max_a_a_oscnt = np.max([train_df['ip_app_oscount'].max(), test_df['ip_app_oscount'].max()])+1
max_i_acnt = np.max([train_df['ip_appcount'].max(), test_df['ip_appcount'].max()])+1
max_i_a_cntuni = np.max([train_df['ip_by_app_countuniq'].max(), test_df['ip_by_app_countuniq'].max()])+1
max_i_c_cntuni = np.max([train_df['ip_by_channel_countuniq'].max(), test_df['ip_by_channel_countuniq'].max()])+1
max_i_d_cntuni = np.max([train_df['ip_by_device_countuniq'].max(), test_df['ip_by_device_countuniq'].max()])+1
max_i_os_cmcnt = np.max([train_df['ip_by_os_cumcount'].max(), test_df['ip_by_os_cumcount'].max()])+1
max_i_c_prvcli = np.max([train_df['ip_channel_prevClick'].max(), test_df['ip_channel_prevClick'].max()])+1
max_i_cnt_c = np.max([train_df['ip_cnt_channel'].max(), test_df['ip_cnt_channel'].max()])+1
max_i_day_h_cntuni = np.max([train_df['ip_day_by_hour_countuniq'].max(), test_df['ip_day_by_hour_countuniq'].max()])+1
max_i_day_hcnt = np.max([train_df['ip_day_hourcount'].max(), test_df['ip_day_hourcount'].max()])+1
max_i_d_os_a_cntuni = np.max([train_df['ip_device_os_by_app_countuniq'].max(), test_df['ip_device_os_by_app_countuniq'].max()])+1
max_i_d_os_a_cmcnt = np.max([train_df['ip_device_os_by_app_cumcount'].max(), test_df['ip_device_os_by_app_cumcount'].max()])+1
max_i_os_d_a_nxtcli = np.max([train_df['ip_os_device_app_nextClick'].max(), test_df['ip_os_device_app_nextClick'].max()])+1
max_i_os_d_nextcli = np.max([train_df['ip_os_device_nextClick'].max(), test_df['ip_os_device_nextClick'].max()])+1
max_i_os_prvcli = np.max([train_df['ip_os_prevClick'].max(), test_df['ip_os_prevClick'].max()])+1

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'app_by_channel_countuniq': np.array(dataset.app_by_channel_countuniq),
        'ip_app_by_os_countuniq': np.array(dataset.ip_app_by_os_countuniq),
        'ip_app_oscount': np.array(dataset.ip_app_oscount),
        'ip_appcount': np.array(dataset.ip_appcount),
        'ip_by_app_countuniq': np.array(dataset.ip_by_app_countuniq),
        'ip_by_channel_countuniq': np.array(dataset.ip_by_channel_countuniq),
        'ip_by_device_countuniq': np.array(dataset.ip_by_device_countuniq),
        'ip_by_os_cumcount': np.array(dataset.ip_by_os_cumcount),
        'ip_cnt_channel': np.array(dataset.ip_cnt_channel),
        'ip_day_by_hour_countuniq': np.array(dataset.ip_day_by_hour_countuniq),
        'ip_day_hourcount': np.array(dataset.ip_day_hourcount),
        'ip_device_os_by_app_countuniq': np.array(dataset.ip_device_os_by_app_countuniq),
        'ip_device_os_by_app_cumcount': np.array(dataset.ip_device_os_by_app_cumcount)

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
in_max_a_c_cntuni = Input(shape=[1], name='app_by_channel_countuniq')
emb_max_a_c_cntuni = Embedding(max_a_c_cntuni, emb_n)(in_max_a_c_cntuni)
in_max_i_a_os_cntuni = Input(shape=[1], name='ip_app_by_os_countuniq')
emb_max_i_a_os_cntuni = Embedding(max_i_a_os_cntuni , emb_n)(in_max_i_a_os_cntuni)
in_max_i_a_d_os_c_nxtcli = Input(shape=[1], name='ip_app_device_os_channel_nextClick')
emb_max_i_a_d_os_c_nxtcli = Embedding(max_i_a_d_os_c_nxtcli, emb_n)(in_max_i_a_d_os_c_nxtcli)
in_max_a_a_oscnt = Input(shape=[1], name='ip_app_oscount')
emb_max_a_a_oscnt = Embedding(max_a_a_oscnt, emb_n)(in_max_a_a_oscnt)
in_max_i_acnt = Input(shape=[1], name='ip_appcount')
emb_max_i_acnt = Embedding(max_i_acnt, emb_n)(in_max_i_acnt)
in_max_i_a_cntuni = Input(shape=[1], name='ip_by_app_countuniq')
emb_max_i_a_cntuni = Embedding(max_i_a_cntuni, emb_n)(in_max_i_a_cntuni)
in_max_i_c_cntuni = Input(shape=[1], name='ip_by_channel_countuniq')
emb_max_i_c_cntuni = Embedding(max_i_c_cntuni, emb_n)(in_max_i_c_cntuni)
in_max_i_d_cntuni = Input(shape=[1], name='ip_by_device_countuniq')
emb_max_i_d_cntuni = Embedding(max_i_d_cntuni, emb_n)(in_max_i_d_cntuni)
in_max_i_os_cmcnt = Input(shape=[1], name='ip_by_os_cumcount')
emb_max_i_os_cmcnt = Embedding(max_i_os_cmcnt, emb_n)(in_max_i_os_cmcnt)
in_max_i_c_prvcli = Input(shape=[1], name='ip_channel_prevClick')
emb_max_i_c_prvcli = Embedding(max_i_c_prvcli, emb_n)(in_max_i_c_prvcli)
in_max_i_cnt_c = Input(shape=[1], name='ip_cnt_channel')
emb_max_i_cnt_c = Embedding(max_i_cnt_c, emb_n)(in_max_i_cnt_c)
in_max_i_day_h_cntuni = Input(shape=[1], name='ip_day_by_hour_countuniq')
emb_max_i_day_h_cntuni = Embedding(max_i_day_h_cntuni, emb_n)(in_max_i_day_h_cntuni)
in_max_i_day_hcnt = Input(shape=[1], name='ip_day_hourcount')
emb_max_i_day_hcnt = Embedding(max_i_day_hcnt, emb_n)(in_max_i_day_hcnt)
in_max_i_d_os_a_cntuni = Input(shape=[1], name='ip_device_os_by_app_countuniq')
emb_max_i_d_os_a_cntuni = Embedding(max_i_d_os_a_cntuni, emb_n)(in_max_i_d_os_a_cntuni)
in_max_i_d_os_a_cmcnt = Input(shape=[1], name='ip_device_os_by_app_cumcount')
emb_max_i_d_os_a_cmcnt = Embedding(max_i_d_os_a_cmcnt, emb_n)(in_max_i_d_os_a_cmcnt)
in_max_i_os_d_a_nxtcli = Input(shape=[1], name='ip_os_device_app_nextClick')
emb_max_i_os_d_a_nxtcli = Embedding(max_i_os_d_a_nxtcli, emb_n)(in_max_i_os_d_a_nxtcli)
in_max_i_os_d_nextcli = Input(shape=[1], name='ip_os_device_nextClick')
emb_max_i_os_d_nextcli = Embedding(max_i_os_d_nextcli, emb_n)(in_max_i_os_d_nextcli)
in_max_i_os_prvcli = Input(shape=[1], name='ip_os_prevClick')
emb_max_i_os_prvcli = Embedding(max_i_os_prvcli, emb_n)(in_max_i_os_prvcli)

fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h),
                (emb_max_i_a_os_cntuni),(emb_max_a_a_oscnt), (emb_max_i_acnt),
                (emb_max_i_a_cntuni), (emb_max_i_c_cntuni), (emb_max_i_d_cntuni),
                (emb_max_i_os_cmcnt), (emb_max_i_cnt_c), (emb_max_i_day_h_cntuni),
                (emb_max_i_day_hcnt), (emb_max_i_d_os_a_cntuni), (emb_max_i_d_os_a_cmcnt)])

s_dout = SpatialDropout1D(0.2)(fe)

x = Flatten()(s_dout)
x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
x = Dropout(0.2)(Dense(dense_n, activation='relu')(x))
outp = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[in_app, in_ch, in_dev, in_os, in_h, in_max_i_a_os_cntuni,
                    in_max_a_a_oscnt, in_max_i_acnt, in_max_i_a_cntuni,
                    in_max_i_c_cntuni, in_max_i_d_cntuni, in_max_i_os_cmcnt,
                    in_max_i_cnt_c, in_max_i_day_h_cntuni, in_max_i_day_hcnt, in_max_i_d_os_a_cntuni,
                    in_max_i_d_os_a_cmcnt], outputs=outp)

batch_size = 20000
epoch = 2
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1))-1
steps = int(len(train_df) / batch_size) * epoch
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizers_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy', optimizer=optimizers_adam, metrics=['accuracy'])

model.summary()

model.fit(train_df, y_train, batch_size=batch_size, epochs=epoch, shuffle=True, verbose=2)
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
sub.to_csv('20180503_dnn_model6.csv', index=False)
