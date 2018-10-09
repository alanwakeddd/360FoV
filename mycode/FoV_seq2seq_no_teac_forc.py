"""
-first part:
seq2seq without teacher forcing
-second part:
seq2seq without teacher forcing, with others' future convLSTM
concat states with decoder LSTM and then predict
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
import sys,glob,io,random
if '/scratch/cl2840/360video/' not in sys.path:
    sys.path.insert(0, '/scratch/cl2840/360video/')
from mycode.dataLayer import DataLayer
import mycode.cost as costfunc
from mycode.config import cfg
from mycode.dataIO import clip_xyz
from mycode.utility import reshape2second_stacks,get_data
from mycode.utility import get_shuffle_index,shuffle_data,get_gt_target_xyz,get_gt_target_xyz_oth
from mycode.utility import slice_layer
from random import shuffle
import matplotlib.pyplot as plt
import _pickle as pickle
import numpy as np
import pdb
### ====================Graph def====================
def onelayer_tar_seq2seq():
    # The first part is unchanged
    if not cfg.input_mean_var:
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
    else:
        encoder_inputs = Input(shape=(None, num_decoder_tokens))    
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]

    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')

    all_outputs = []
    inputs = decoder_inputs
    for _ in range(max_decoder_seq_length):
        # Run the decoder on one timestep
        decoder_states, state_h, state_c = decoder_lstm(inputs,
                                                 initial_state=states)
        outputs = decoder_dense(decoder_states)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # Define and compile model as previously
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


if __name__ == '__main__':
    experiment = 1
    batch_size = 32  # Batch size for training.
    epochs = 200  # Number of epochs to train for.
    latent_dim = 64  # Latent dimensionality of the encoding space.

    fps = 30
    num_encoder_tokens = 3*fps
    num_decoder_tokens = 6
    max_encoder_seq_length = cfg.running_length
    max_decoder_seq_length = cfg.predict_step
    # num_user = 48


    model = onelayer_tar_seq2seq()
    model.compile(optimizer='Adam', loss='mean_squared_error')



    #### ====================data====================
    ## load data just as in Fov_seq2seq.py
    # if cfg.use_xyz:
    #     all_video_data = pickle.load(open('/scratch/cl2840/360video/data/new_exp_'+str(experiment)+'_xyz.p','rb'))
    #     data_dim = 3
    # all_video_data = clip_xyz(all_video_data)
    # datadb = all_video_data.copy()
    ##### data format 3or4--4
    # video_data_train = pickle.load(open('/scratch/cl2840/360video/data/shanghai_dataset_xyz_train.p','rb'))    
    #### format5
    video_data_train = pickle.load(open('/scratch/cl2840/360video/temp/tsinghua_after_bysec_interpolation/tsinghua_train_video_data_over_video.p','rb')) 
    video_data_train = clip_xyz(video_data_train)
    datadb = video_data_train.copy()


    _video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)
    # total_num_samples = _video_db.shape[0]

    if cfg.shuffle_data:
        #shuffle the whole dataset
        # index_shuf = get_shuffle_index(total_num_samples)
        index_shuf = pickle.load(open('index_shuf'+'_exp'+str(experiment)+'.p','rb'))
        _video_db = shuffle_data(index_shuf,_video_db)
        _video_db_future = shuffle_data(index_shuf,_video_db_future)
        _video_db_future_input = shuffle_data(index_shuf,_video_db_future_input)


    # num_testing_sample = int(0.15*total_num_samples)#use last few as test
    num_testing_sample = 1
    if cfg.input_mean_var:
        encoder_input_data = get_gt_target_xyz(_video_db[:-num_testing_sample,:,:])
    else:
        encoder_input_data = _video_db[:-num_testing_sample,:,:]
    decoder_target_data = get_gt_target_xyz(_video_db_future)[:-num_testing_sample,:,:]
    # decoder_input_data = get_gt_target_xyz(_video_db_future_input)[:-num_testing_sample,-1,:][:,np.newaxis,:]
    decoder_input_data = get_gt_target_xyz(_video_db_future_input)[:-num_testing_sample,0,:][:,np.newaxis,:]



    ### ====================Training====================
    # model = load_model('fov_s2s_noteacherforcing_epoch21-0.0759.h5')
    # tag = 'notFor_tanh_newdata_exp2_epoch'
    # tag = 'fctar_seqseq_shanghai_traintest_split_predmeanvar_Aug9'
    # tag = 'fctar_seqseq_shanghai_traintest_split_meanvarmeanvar_Aug9'
    tag = 'fctar_seqseq_THU_predmeanvar_Sep5'
    model_checkpoint = ModelCheckpoint(tag+'{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                     patience=3, min_lr=1e-6)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True,initial_epoch=0,
              callbacks=[model_checkpoint, reduce_lr, stopping])



    ### ====================Testing====================
    ##### data format 3or4--4
    # video_data_test = pickle.load(open('/scratch/cl2840/360video/data/shanghai_dataset_xyz_test.p','rb'))
    ### data format 5
    video_data_test = pickle.load(open('/scratch/cl2840/360video/temp/tsinghua_after_bysec_interpolation/tsinghua_test_video_data_over_video.p','rb'),encoding='latin1')

    video_data_test = clip_xyz(video_data_test)
    datadb = video_data_test.copy()
    _video_db,_video_db_future,_video_db_future_input = get_data(datadb,pick_user=False)

    if cfg.input_mean_var:
        _video_db = get_gt_target_xyz(_video_db)

    # model = load_model('fov_s2s_noteacherforcing.h5')
    # model = load_model('fov_s2s_noteacherforcing_epoch74-0.0087.h5')
    # model = load_model('notFor_tanh_newdata_epoch18-0.0761.h5')
    # model = load_model('notFor_tanh_newdata_exp2_epoch200-0.0302.h5')
    def decode_sequence_fov(input_seq):
        # Encode the input as state vectors.
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
        if not cfg.input_mean_var:
            last_mu_var = get_gt_target_xyz(last_location)
        else:
            last_mu_var = last_location
        decoded_sentence = model.predict([input_seq,last_mu_var])
        return decoded_sentence


    gt_sentence_list = []
    decoded_sentence_list = []
    # for seq_index in range(total_num_samples-num_testing_sample,total_num_samples):
    for seq_index in range(_video_db.shape[0]):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = _video_db[seq_index: seq_index + 1,:,:]

        decoded_sentence = decode_sequence_fov(input_seq)
        decoded_sentence_list+=[decoded_sentence]
        gt_sentence = _video_db_future[seq_index: seq_index + 1,:,:]
        gt_sentence_list+=[gt_sentence]
        decoder_target = get_gt_target_xyz(gt_sentence)
        # print('-')
        # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

    pickle.dump(decoded_sentence_list,open('decoded_sentence'+tag+'.p','wb'))
    pickle.dump(gt_sentence_list,open('gt_sentence_list'+tag+'.p','wb'))
    print('Testing finished!')












    ### ==============Below is concating with others' future branch convLSTM===========================
    ## utility layers
    flatten_layer = Flatten()
    expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
    Concatenatelayer = Concatenate(axis=1)
    get_dim1_layer = Lambda(lambda x: x[:,0,:])


    ### ====================Graph def====================
    # The first part is unchanged
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = [state_h, state_c]


    ###======convLSTM on others' future======
    other_fut_lstm = ConvLSTM2D(filters=latent_dim, kernel_size=(num_user-1, 3),
                       input_shape=(1, num_user-1, fps, 3),
                       padding='same', return_sequences=True, return_state=True)
    others_fut_inputs = Input(shape=(max_decoder_seq_length,num_user-1,fps,3))
    # others_fut_inputs = Input(batch_shape=(batch_size,1,num_user-1,fps,3))
    flatten_conv_lstm_state_dense = Dense(latent_dim)

    fut_outputs_sqns, others_state_h, others_state_c = other_fut_lstm(others_fut_inputs)




    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    # decoder_inputs = Input(batch_shape=(batch_size, 1, num_decoder_tokens))
    # decoder_lstm = LSTM(latent_dim, return_sequences=False, return_state=True, stateful=True)
    decoder_dense = Dense(num_decoder_tokens,activation='tanh')


    ## concat states
    all_outputs = []
    inputs = decoder_inputs
    for time_ind in range(max_decoder_seq_length):
        # Run the decoder on one timestep
        decoder_states, state_h, state_c = decoder_lstm(inputs,initial_state=states)

        # ### caution: it seems keras convLSTM is by default stateful, don't have to feed back last hidden states.
        # ### is this true?
        # fut_outputs, others_state_h, others_state_c = other_fut_lstm(others_fut_inputs)
        # # fut_outputs, others_state_h, others_state_c = other_fut_lstm(others_fut_inputs,initial_state=others_states)#erros?!!!
        
        # fut_outputs = identity_layer(fut_outputs_sqns[:,time_ind,:,:,:])
        fut_outputs = slice_layer(1,time_ind,time_ind+1)(fut_outputs_sqns)
        convlstm_state = flatten_layer(fut_outputs)
        convlstm_state = flatten_conv_lstm_state_dense(convlstm_state)
        concat_state = Concatenatelayer([get_dim1_layer(decoder_states),convlstm_state])
        outputs = decoder_dense(concat_state)
        outputs = expand_dim_layer(outputs)
        all_outputs.append(outputs)

        inputs = outputs
        states = [state_h, state_c]

        # others_fut_inputs = others_fut_inputs #TODO feed gt others for next step
        # others_states = [others_state_h, others_state_c]


    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)


    # Define and compile model as previously
    model = Model([encoder_inputs, others_fut_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='Adam', loss='mean_squared_error')




    #### ====================data====================
    # load cached data
    _video_db_tar = pickle.load(open('./cache/_video_db_tar_exp'+str(experiment)+'.p','rb'))
    _video_db_future_tar = pickle.load(open('./cache/_video_db_future_tar_exp'+str(experiment)+'.p','rb'))
    _video_db_future_input_tar = pickle.load(open('./cache/_video_db_future_input_tar_exp'+str(experiment)+'.p','rb'))
    _video_db_oth = pickle.load(open('./cache/_video_db_oth_exp'+str(experiment)+'.p','rb'))
    _video_db_future_oth = pickle.load(open('./cache/_video_db_future_oth_exp'+str(experiment)+'.p','rb'))
    _video_db_future_input_oth = pickle.load(open('./cache/_video_db_future_input_oth_exp'+str(experiment)+'.p','rb'))




    def data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar):
        sample_ind = np.random.randint(0,_video_db_future_input_tar.shape[0])
        assert (_video_db_tar[sample_ind,:][-1,:]-_video_db_future_input_tar[sample_ind,:][0,:]).sum()==0
        print(np.abs(_video_db_tar[sample_ind,:][-1,:]-_video_db_future_tar[sample_ind,:][0,:]))
        
    def _reshape_others_data(_video_db_oth):
        ## to match Input shape: others_fut_inputs
        _video_db_oth = _video_db_oth.transpose((1,2,0,3))
        _video_db_oth = _video_db_oth.reshape((_video_db_oth.shape[0],_video_db_oth.shape[1],_video_db_oth.shape[2],
                        fps,3))
        return _video_db_oth

    _video_db_oth = _reshape_others_data(_video_db_oth)
    _video_db_future_oth = _reshape_others_data(_video_db_future_oth)
    # _video_db_future_input_oth = _reshape_others_data(_video_db_future_input_oth)
    total_num_samples = _video_db_tar.shape[0]

    #### shuffle the whole dataset
    # index_shuf = get_shuffle_index(total_num_samples)
    index_shuf = pickle.load(open('index_shuf'+'_exp'+str(experiment)+'.p','rb'))
    print('Shuffle data before training and testing.')
    _video_db_tar = shuffle_data(index_shuf,_video_db_tar)
    _video_db_future_tar = shuffle_data(index_shuf,_video_db_future_tar)
    _video_db_future_input_tar = shuffle_data(index_shuf,_video_db_future_input_tar)

    # _video_db_oth = shuffle_data(index_shuf,_video_db_oth)
    _video_db_future_oth = shuffle_data(index_shuf,_video_db_future_oth)
    # _video_db_future_input_oth = shuffle_data(index_shuf,_video_db_future_input_oth)


    #### prepare training data
    data_sanity_check(_video_db_tar,_video_db_future_tar,_video_db_future_input_tar)
    num_testing_sample = int(0.15*total_num_samples)#use last few as test
    encoder_input_data = _video_db_tar[:-num_testing_sample,:,:]
    decoder_target_data = get_gt_target_xyz(_video_db_future_tar)[:-num_testing_sample,:,:]
    # decoder_input_data = get_gt_target_xyz(_video_db_future_input_tar)[:-num_testing_sample,-1,:][:,np.newaxis,:]
    decoder_input_data = get_gt_target_xyz(_video_db_future_input_tar)[:-num_testing_sample,0,:][:,np.newaxis,:]

    others_fut_input_data = _video_db_future_oth[:-num_testing_sample,:]





    ### ====================Training====================
    # model = load_model('fov_s2s_notfor_concatConvLSTM_epoch82-0.0168.h5')
    # model = load_model('fov_s2s_noTfor_concatConvLSTM_48user_shuffle_epoch10-0.0487.h5')
    # model = load_model('fov_s2s_noTfor_concatConvLSTM_48user_shuffle_epoch06-0.0155.h5')
    # model = load_model('fov_s2s_noTfor_concatConvLSTM_48user_shuffle_epoch26-0.0096.h5')
    model_checkpoint = ModelCheckpoint('concat_future_ConvLSTM_newdata_epoch{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                     patience=3, min_lr=1e-6)
    stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    # Train model as previously 
    model.fit([encoder_input_data, others_fut_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True, initial_epoch=0,
              callbacks=[model_checkpoint, reduce_lr, stopping])






    ### ====================Testing====================
    # model = load_model('fov_s2s_notfor_concatConvLSTM_epoch82-0.0168.h5')
    # model = load_model('concat_future_ConvLSTM_newdata_epoch08-0.8951.h5')
    def decode_sequence_fov(input_seq,others_fut_input_seq):
        # Encode the input as state vectors.
        last_location = input_seq[0,-1,:][np.newaxis,np.newaxis,:]
        last_mu_var = get_gt_target_xyz(last_location)
        decoded_sentence = model.predict([input_seq,others_fut_input_seq,last_mu_var])
        return decoded_sentence

    gt_sentence_list = []
    decoded_sentence_list = []
    for seq_index in range(total_num_samples-num_testing_sample,total_num_samples):
        input_seq = _video_db_tar[seq_index: seq_index + 1,:,:]
        others_fut_input_seq = _video_db_future_oth[seq_index: seq_index + 1,:]
        decoded_sentence = decode_sequence_fov(input_seq,others_fut_input_seq)
        decoded_sentence_list+=[decoded_sentence]
        gt_sentence = _video_db_future_tar[seq_index: seq_index + 1,:,:]
        gt_sentence_list+=[gt_sentence]
        # print('-')
        # decoder_target = get_gt_target_xyz(gt_sentence)
        # print('Decoded sentence - decoder_target:', np.squeeze(np.array(decoded_sentence))[:,:3]-np.squeeze(decoder_target)[:,:3])

    pickle.dump(decoded_sentence_list,open('decoded_sentence.p','wb'))
    pickle.dump(gt_sentence_list,open('gt_sentence_list.p','wb'))
    print('Testing finished!')











    ####TimeDistributed

    # fut_outputs, others_state_h, others_state_c = other_fut_lstm(others_fut_inputs)
    # # convlstm_state = flatten_conv_lstm_state_dense(K.reshape(fut_outputs,(-1,max_decoder_seq_length,(num_user-1)*fps*3)))

    # td_layer = TimeDistributed(flatten_conv_lstm_state_dense)
    # # convlstm_state = td_layer(K.reshape(fut_outputs,(-1,max_decoder_seq_length,(num_user-1)*fps*3)))
    # convlstm_state = flatten_layer(fut_outputs)



    ####!!check compatibility, EVERY layer needs to be keras tensor (especially concat, slice, backend ops)

    # model1 = Model(encoder_inputs, encoder_outputs)

    # model2 = Model(others_fut_inputs, convlstm_state)
    # model5 = Model([encoder_inputs,others_fut_inputs,decoder_inputs], concat_state)
    # model5 = Model([encoder_inputs,others_fut_inputs,decoder_inputs], outputs)
    # model5 = Model([encoder_inputs,others_fut_inputs,decoder_inputs], decoder_outputs)


    # model3 = Model([decoder_inputs,encoder_inputs], decoder_states)
    # model3 = Model([decoder_inputs,encoder_inputs], state_h)




