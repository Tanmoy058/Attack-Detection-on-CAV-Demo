#tensorflow and keras imports
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, Lambda, Concatenate
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Reshape
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras import regularizers
import keras.backend as K
from keras.losses import mse, binary_crossentropy, categorical_crossentropy

import numpy as np
import math
import os 
from sklearn.metrics import mean_squared_error, mean_absolute_error

#imports from python files 
from utils.generic_utils import load_dataset_at
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST 

DATASET_INDEX = 6
MAX_NB_VARIABLE = MAX_NB_VARIABLES[DATASET_INDEX]
MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]

#######################################BASE MODEL###############################################
def build_base_model(weights = None):
    #Squeeze excitation block
    def squeeze_excite_block(input):
        filters = input._keras_shape[-1] # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se


    ip = Input(shape=(MAX_NB_VARIABLE, MAX_TIMESTEPS))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.2)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(64, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 4, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASSES_LIST[0], activation='softmax')(x)

    model = Model(inputs=ip, outputs=out)
    
    model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    
    if(not weights == None):
        model.load_weights(weights)

    return model

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


################################# AUTO ENCODER ###############################################
def build_encoder_decoder_model(input_shape, latent_dims):
    timesteps = input_shape[0]
    encoded_seq_len = input_shape[1]

    
    ################################# AUTO ENCODER ###############################################
    #Encoder
    input_dim = input_shape[0] * input_shape[1]
    input_shape = (input_dim, )
    intermediate_dim = 256
    intermediate_dim_2 = 128
    intermediate_dim_3 = 64
    batch_size = 64
    latent_dim = 3549
    epochs = 40

    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    x_2 = Dense(intermediate_dim_2, activation='relu')(x)
    x_3 = Dense(intermediate_dim_3, activation='relu')(x_2)
    z_mean = Dense(latent_dim, name='z_mean')(x_3)
    z_log_var = Dense(latent_dim, name='z_log_var')(x_3)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x_2 = Dense(intermediate_dim_2, activation='relu')(x)
    x_3 = Dense(intermediate_dim_3, activation='relu')(x_2)
    outputs = Dense(88)(x_3)
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    vae.compile(loss=categorical_crossentropy, optimizer='adam')

    return vae, decoder

################################# Reading and Normalizing data ############################################
X_train, Y_train, X_test, Y_test, is_timeseries = load_dataset_at(DATASET_INDEX, fold_index = None, normalize_timeseries = False) 

#Normalize
train_mean = np.mean(X_train)
train_std = np.std(X_train)

norm_X_train = (X_train - train_mean) / (train_std)
norm_X_test = (X_test - train_mean) / (train_std)

X_test = pad_sequences(X_test, maxlen = MAX_NB_VARIABLE, padding='post', truncating='post')

Y_train = to_categorical(Y_train, len(np.unique(Y_train)))
Y_test = to_categorical(Y_test, len(np.unique(Y_test)))

init_decoder_input = np.zeros(shape=(X_train.shape[0], 1, X_train.shape[2])) #(batch, 1, length_of_sequence)

np.min(norm_X_train), np.max(norm_X_train)

################################## Building Models #############################################
autoencoder, decoder_model = build_encoder_decoder_model(X_train.shape[1:], 256) #only interested in the decoder model part
base_model = build_base_model()

tf.keras.backend.get_session().run(tf.global_variables_initializer())

################################## Train Models ########################################
#history1=autoencoder.fit(x=[X_train, init_decoder_input], y=[X_train], batch_size=32, epochs=100, 
#                        validation_data=[[X_test, init_decoder_input[:len(X_test)]], X_test], 
#                        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=5)])

#history2=base_model.fit(x=norm_X_train, y=Y_train, epochs=50)

################################## Save trained weights ######################################
#decoder_model.save_weights('decoder_weights.h5')
#base_model.save_weights('base_model_weights.h5')

################################## Initialize tensorflow session #####################################
#Initilize shapes of the varaibels
latent_vector_shape = (256,)
X_shape = X_train.shape[1:]
Y_shape = Y_train.shape[1:]

#MAKE graph
sess_autoZoom = tf.InteractiveSession()
k = 0.00
CONST_LAMBDA = tf.placeholder(tf.float32, name='lambda')
x0 = tf.placeholder(tf.float32, (None,) + X_shape, name='x0') #Input data
t0 = tf.placeholder(tf.float32, (None,) + Y_shape, name='t0') #Output

latent_adv = tf.placeholder(tf.float32, (None,) + latent_vector_shape, name='adv') #avdersarial example
init_dec_in = tf.placeholder(tf.float32, (None, 1, X_shape[1]), name ='Dec')

# compute loss
decoder_model.load_weights('decoder_weights.h5')
adv = decoder_model([init_dec_in, latent_adv])

# Make sure that adversarial example is not too far from the X0, can comment out this part if problem arises
adv_up = 10 + x0
adv_down = -10 - x0
cond1 = tf.cast(tf.greater(adv, adv_up), tf.float32)
cond2 = tf.cast(tf.less_equal(adv, adv_up), tf.float32)
cond3 = tf.cast(tf.greater(adv, adv_down), tf.float32)
cond4 = tf.cast(tf.less_equal(adv, adv_down), tf.float32)
adv = tf.multiply(cond1, adv_up) + tf.multiply(tf.multiply(cond2, cond3), adv) + tf.multiply(cond4, adv_down)

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
base_model.load_weights('./base_model_weights.h5')
decoder_model.load_weights('./decoder_weights.h5')



#Optimization 
success_count = 0
summary = {'l2_loss': {}, 'adv': {}, 'query': {}, 'l0_loss': {}, 'iter': {}}
fail_count = 0
invalid_count = 0
S = 10
init_lambda = 1000

grad = np.zeros((1, latent_vector_shape[0]), dtype = np.float32)
#Iterate for each test example
for i in range(norm_X_test.shape[0]):

    print("\n start attacking target", i, "...")
       
    mt = 0.0           # accumulator m_t in Adam
    vt = 0.0           # accumulator v_t in Adam

    beta1 = 0.9            # parameter beta_1 in Adam
    beta2 = 0.999          # parameter beta_2 in Adam
    learning_rate = 2e-3           # learning rate in Adam
    
    batch_size = 1                # batch size
    Max_Query_count = 2000        # maximum number of queries allowed

    best_l2 = np.math.inf         #setting best l2 score to infinity

    #initial latent sapce adversarial input
    init_adv = np.zeros((1,) + latent_vector_shape)           # initial adversarial perturbation

    X = np.expand_dims(norm_X_test[i], 0)      # target sample X (normalized)
    Y = np.expand_dims(Y_test[i], 0)           # target sample's lable Y
    
    # check if (X, Y) is a valid target, y checking if it is classified correctly
    Y_pred = base_model.predict(X)
    if sum((max(Y_pred[0]) == Y_pred[0]) * Y[0]) == 0:
        print("not a valid target.")
        invalid_count += 1
        continue

    #dummy Decoder intial input just an array of 0's
    X2 = np.zeros((1, 1, MAX_TIMESTEPS))

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
            q = 7 
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

        #For estimation of F(x + beta*u) and F(x) x = init_adversarial
        var = np.concatenate((init_adv, init_adv + beta * u.reshape((q,)+ (latent_vector_shape))), axis=0)
        
        l2_loss, losses, scores = sess_autoZoom.run([Dist, f, t], feed_dict={latent_adv: var, x0: X, t0: Y, init_dec_in: X2, 
                                                                            CONST_LAMBDA: init_lambda}) 

        #Gradient estimation
        for j in range(q):
            grad[j] = (b * (losses[j + 1] - losses[0])* u[j]) / beta
        
        avg_grad = np.mean(grad, axis=0)

        # ADAM update
        mt = beta1 * mt + (1 - beta1) * avg_grad
        vt = beta2 * vt + (1 - beta2) * (avg_grad * avg_grad)
        corr = (np.sqrt(1 - np.power(beta2, epoch))) / (1 - np.power(beta1, epoch))

        m = init_adv.reshape(-1)
        m -= learning_rate * corr * mt / (np.sqrt(vt) + 1e-8)

        #update the adversarial example
        init_adv = m.reshape(init_adv.shape)
        
        l2_loss = np.mean(l2_loss)
        if(sum((scores[0] == max(scores[0]))*Y[0]) !=0 and epoch%S == 0):
            init_lambda /= 2

        if(sum((scores[0] == max(scores[0]))*Y[0])==0 and l2_loss < best_l2):
            if(np.math.isinf(best_l2)):
                print("Initial attack found on query {query} and l2 loss of {l2_loss}")
            best_l2 = l2_loss
            summary['l2_loss'][i] = l2_loss
            summary['query'][i] = query
            summary['adv'][i] = init_adv          
        
        if(query >= Max_Query_count and not np.math.isinf(best_l2)):
            print("Attack succeeded! with best l2 loss: {summary['l2_loss'][i]} and query count: {summary['query'][i]}")
            success_count += 1
        
        elif (query >= Max_Query_count and np.math.isinf(best_l2)):
            print("attack failed!")
            fail_count += 1
            break

a = []
for i in summary['query'].keys():
    if summary['query'][i] != 500:
        a.append(sum(sum(abs(summary['adv'][i][0][:,:]) > 0)) /2/ max(sum(abs(X_test[i][0])>0), sum(abs(X_test[i][1])>0), 1))

avg_l0 = sum(a) / len(a)

res_summary = {"success_count": success_count, \
        "fail_count": fail_count, \
        "invalid_count": invalid_count, \
        "target_model_accuracy": 1-invalid_count/X_test.shape[0], \
        "success_rate": success_count/(fail_count+success_count), \
        "avg_ite": sum( [x for x in list(summary['query'].values()) if x!=500])/success_count, \
        "avg_que": sum( [x for x in list(summary['query'].values()) if x!=500])/(success_count + fail_count), \
        "avg_l2": sum( [x for x in list(summary['l2_loss'].values()) if x!=500])/success_count, \
        "avg_l0": avg_l0
        }

print("\n-------operation result-------\n")
print("Successful times: {success_count}, \n\
number of failures: {fail_count}, \n\
Target model accuracy: {target_model_accuracy}, \n\
attack Success rate: {success_rate}, \n\
Average disturbancel2: {avg_l2_}, \n\
Average disturbance l2 ratio: {avg_l2}, \n\
Average disturbance l0 ratio: {avg_l0}, \n\
Average number of iterations: {avg_ite}".format(**res_summary))