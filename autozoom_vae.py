import numpy as np
import math
import os 

import tensorflow as tf
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau as ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint as ModelCheckpoint
import keras.backend as K
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.models import Model

from models.AE import create_rnn_ae
from models.VAE import create_rnn_vae
from models.model import create_classification_model

from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST 
from utils.keras_utils import train_model
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 0


MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASSES_LIST = NB_CLASSES_LIST[DATASET_INDEX]
MAX_TIMESTEPS_LIST = MAX_TIMESTEPS_LIST[DATASET_INDEX]

X_train, Y_train, X_test, Y_test, is_timeseries = load_dataset_at(DATASET_INDEX, fold_index = None, normalize_timeseries = True) 


X_test = pad_sequences(X_test, maxlen = MAX_NB_VARIABLES, padding='post', truncating='post')

Y_train = to_categorical(Y_train, len(np.unique(Y_train)))
Y_test = to_categorical(Y_test, len(np.unique(Y_test)))


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

init_decoder_input = np.zeros(shape=(X_train.shape[0], 1, X_train.shape[2])) #(batch, 1, length_of_sequence)

np.min(X_test), np.max(X_test)

def generate_MLSTM_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_MLSTM_attention_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))
    # stride = 10

    # x = Permute((2, 1))(ip)
    # x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
    #            kernel_initializer='he_uniform')(x)  # (None, variables / stride, timesteps)
    # x = Permute((2, 1))(x)

    #ip1 = K.reshape(ip,shape=(MAX_TIMESTEPS,MAX_NB_VARIABLES))
    #x = Permute((2, 1))(ip)
    x = Masking()(ip)
    x = AttentionLSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model

def generate_FCN_model():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST))

    #x = Masking()(ip)
    #x = LSTM(8)(x)
    #x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    #x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST, activation='softmax')(y)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model
    
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

vae, encoder_model, decoder_model = create_rnn_vae(MAX_NB_VARIABLES, MAX_TIMESTEPS_LIST, None, 256, 128) 
base_model = generate_MLSTM_attention_model()

train_model(base_model, DATASET_INDEX, dataset_prefix='MLSTM_attention', epochs=40, batch_size=16, normalize_timeseries=True, monitor="val_accuracy", 
            optimization_mode="max")

base_model.save_weights('./classification_model_5.h5')

#train autoencoder
history1=vae.fit(x=[X_train], y=[X_train], batch_size=32, epochs=40, 
                        validation_data=[X_test , X_test], 
                        callbacks=[ReduceLROnPlateau(patience=5), 
                        ModelCheckpoint('vae.h5',save_best_only=True,save_weights_only=True)])

decoder_model.save_weights('vae_decoder.h5')

#MAKE graph .transpose(0,2,1)
sess_autoZoom = tf.InteractiveSession()
'''
    Construct tf-Graph
''' 
#latent_vector_shape = (MAX_TIMESTEPS_LIST,)
latent_vector_shape = (128,)
X_shape = X_train.shape[1:]
Y_shape = Y_train.shape[1:]

k = 0.00
CONST_LAMBDA = tf.placeholder(tf.float32, name='lambda')
x0 = tf.placeholder(tf.float32, (None,) + X_shape, name='x0') #Input data
t0 = tf.placeholder(tf.float32, (None,) + Y_shape, name='t0') #Output

latent_adv = tf.placeholder(tf.float32, (None,) + latent_vector_shape, name='adv') #avdersarial example
init_dec_in = tf.placeholder(tf.float32, (None, 1, X_shape[1]), name ='Dec')

# compute loss
adv = decoder_model(latent_adv)
#adv = np.zeros((1,) + latent_vector_shape)
print(adv.shape)
print(x0.shape)

x = adv + x0
t = base_model(x)

Dist = tf.reduce_sum(tf.square(x - x0), axis=[1,2])

real = tf.reduce_sum(t0 * t, axis=1)
other = tf.reduce_max((1 - t0) * t - t0*10000, axis=1)

#untargeted attack    
Loss = CONST_LAMBDA * tf.maximum(tf.log(real + 1e-30) - tf.log(other + 1e-30), -k)

f = Dist + Loss
# # initialize variables and load target model
sess_autoZoom.run(tf.global_variables_initializer())

#weights are reset
base_model.load_weights('./classification_model_5.h5')
decoder_model.load_weights('./vae_decoder.h5')

success_count = 0
summary = {'init_l0': {}, 'init_l2': {}, 'l0': {}, 'l2': {}, 'adv': {}, 'query': {}, 'epoch': {}}
fail_count, invalid_count = 0, 0
S = 100
init_lambda = 10000

grad = np.zeros((1, latent_vector_shape[0]), dtype = np.float32)

#Iterate for each test example
for i in range(X_test.shape[0]):

    print("\n start attacking target", i, "...")
       
    mt = 0           # accumulator m_t in Adam
    vt = 0           # accumulator v_t in Adam

    beta1 = 0.9            # parameter beta_1 in Adam
    beta2 = 0.999          # parameter beta_2 in Adam
    learning_rate = 2e-3          # learning rate in Adam
    
    batch_size = 1                # batch size
    Max_Query_count = 2000        # maximum number of queries allowed

    best_l2 = np.math.inf

    #For the time being it has the same shape as X
    init_adv = np.zeros((1,) + latent_vector_shape)           # initial adversarial perturbation

    X = np.expand_dims(X_test[i], 0)           # target sample X
    Y = np.expand_dims(Y_test[i], 0)           # target sample's lable Y
    
    # check if (X, Y) is a valid target, y checking if it is classified correctly
    Y_pred = base_model.predict(X)
    if sum((max(Y_pred[0]) == Y_pred[0]) * Y[0]) == 0:
        #print("not a valid target.")
        invalid_count += 1
        continue

    var_size = init_adv.size
    beta = 1/(var_size)

    query, epoch = 0, 0
    q = 1 
    b = q
    # main loop for the optimization
    while(query < Max_Query_count):
        epoch += 1
        #if initial attack is found fine tune the adversarial example buy increasing the q
        if(not np.math.isinf(best_l2)):
            q = 3 
            b = q
            grad = np.zeros((q, var_size), dtype = np.float32)

        query += q #q queries will be made in this iteration
        
        #Using random vector gradient estimation 

        #random noise
        u = np.random.normal(loc=0, scale=1000, size = (q, var_size))
        u_mean = np.mean(u, axis=1, keepdims=True)
        u_std = np.std(u, axis=1, keepdims=True)
        u_norm = np.apply_along_axis(np.linalg.norm, 1, u, keepdims=True)
        u = u/u_norm

        #For estimation of F(x + beta*u) and F(x)
        var = np.concatenate((init_adv, init_adv + beta * u.reshape((q,)+ (latent_vector_shape))), axis=0)
        
        l0_loss, l2_loss, losses, scores = sess_autoZoom.run([Loss, Dist, f, t], feed_dict={latent_adv: var, x0: X, t0: Y, 
                                                                            CONST_LAMBDA: init_lambda}) 

        #Gradient estimation
        for j in range(q):
            if len(losses) > 1:
                grad[j] = (b * (losses[j + 1] - losses[0])* u[j]) / beta
            else:
                grad[j] = (b * (losses[0])* u[j]) / beta
            
        avg_grad = np.mean(grad, axis=0)

        # ADAM update
        mt = beta1 * mt + (1 - beta1) * avg_grad
        vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
        corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))

        m = init_adv.reshape(-1)
        m -= learning_rate * corr * mt / (np.sqrt(vt) + 1e-8)

        #update the adversarial example
        init_adv = m.reshape(init_adv.shape)
        
        l2_loss = l2_loss[0]
        l0_loss = l0_loss[0]
        
        if(epoch%S == 0 and not np.math.isinf(best_l2)):
            init_lambda /= 2

        if(sum((scores[0] == max(scores[0]))*Y[0])==0 and l2_loss < best_l2):
           
            if(np.math.isinf(best_l2)):
                #print("Initial attack found on query {query} and l2 loss of {l2_loss}")
                summary['query'][i] = query
                summary['epoch'][i] = epoch
                summary['init_l0'][i] = l0_loss
                summary['init_l2'][i] = l2_loss

            best_l2 = l2_loss
            summary['l0'][i] = l0_loss
            summary['l2'][i] = l2_loss
            summary['adv'][i] = init_adv            
        
        if(query >= Max_Query_count and not np.math.isinf(best_l2)):
            #print("Attack successed! with best l2 loss:")
            #print(summary['l2'][i])
            success_count += 1

        elif (query >= Max_Query_count and np.math.isinf(best_l2)):
            #print("attack failed!")
            fail_count += 1
            break

print(invalid_count, fail_count)
print(len(summary['adv'].keys()) /(len(summary['adv'].keys())+ fail_count))
print(np.average(list(summary['query'].values())))
print(np.average(list(summary['epoch'].values())))
print(np.average(list(summary['init_l2'].values())))
print(np.average(list(summary['init_l0'].values())))
print(np.average(list(summary['l2'].values())))
print(np.average(list(summary['l0'].values())))


print("Invalid Count", invalid_count, "Fail Count", fail_count)
print("Success Rate:", (1 - (fail_count/X_test.shape[0])))
print(len(summary['adv'].keys()) /(len(summary['adv'].keys())+ fail_count))
print("Query Average", np.average(list(summary['query'].values())))
print("Iter Average", np.average(list(summary['epoch'].values())))
print("L2 Average:", np.average(list(summary['init_l2'].values())))
print("L0 Average:", np.average(list(summary['init_l0'].values())))
print("L2 Ratio Average:", np.average(list(summary['l2'].values())))

a = []
for i in summary['query'].keys():
    if summary['query'][i] != 500:
        a.append(sum(sum(abs(summary['adv'][i][0][:,:]) > 0)) /2/ max(sum(abs(X_test[i][0])>0), sum(abs(X_test[i][1])>0), 1))

avg_l0 = sum(a) / len(a)
print("L0 Ratio Average:", avg_l0)
