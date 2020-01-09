import re
import os
import argparse
import shutil
import time
import json
import numpy as np
import pandas as pd
from glob import glob

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from data_utils import getTrainXY, json_out

###############################################################

def getModel(name, height, width):
    if name == 'STResNet':
        from STResNet import st_resnet
        return st_resnet(close_input=(height, width, channel*timesteps))
    elif name == 'ConvLSTM':
        from ConvLSTM import convlstm
        return convlstm(seq_input=(timesteps, height, width, channel))
    else:
        raise Exception('Not a valid model name.')


def trainModel(model, modelName, trainX, trainY, tempPath):
    train_flag = tempPath + '/finish.txt'
    if os.path.exists(train_flag):
        os.remove(train_flag)    # no 'finish.txt' during training

    # XY features
    if modelName == 'STResNet':
        # transpose XS
        XS = np.expand_dims(trainX, axis=-2)
        XS = np.squeeze(XS.swapaxes(-2, 1))
        XS = XS.reshape((XS.shape[0], XS.shape[1], XS.shape[2], -1))
        YS = trainY
    elif modelName == 'ConvLSTM':
        XS, YS = trainX, trainY
    print(XS.shape, YS.shape)
    print('Model Training Started ...', time.ctime())

    csv_logger = CSVLogger(tempPath + '/' + modelName + '.log')
    checkpointer = ModelCheckpoint(tempPath + '/' + modelName + '.h5', verbose=1, save_best_only=True)
    LR = LearningRateScheduler(lambda epoch: LEARN)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True,
              callbacks=[csv_logger, checkpointer, LR], validation_split=SPLIT)

    print('Model Training Ended ...', time.ctime())
    f = open(train_flag, 'w')     # indicating end of training
    f.close()

    return None


def testModel(model, modelName, height, width, pmax, which_type, user_name, testPath, outputPath, tempPath, backPath):
    print('Model Test Started ...', time.ctime())

    assert os.path.exists(tempPath + '/' + modelName + '.h5'), 'model is not existing'
    model.load_weights(tempPath + '/'+ modelName + '.h5')
    model.summary()

    XS = []
    watcher = 0   # length of XS input
    t = pd.Timedelta(30, 'm')
    last_t = pd.to_datetime(time.ctime(), format = '%a %b %d %H:%M:%S %Y')   # initiate new_t
    
    while True:
        cur_t = pd.to_datetime(time.ctime(), format = '%a %b %d %H:%M:%S %Y')
        if cur_t - last_t >= t:
            print('Waited longer than 30min. Quit..')
            break
        
        for file in sorted(glob(os.path.join(testPath, '*'+user_name+'.json'))):
            filename = file.split('/')[-1]
            if re.match(r'^\.', filename) is None:   # able to read a file (not hidden)
                last_t = pd.to_datetime(time.ctime(), format = '%a %b %d %H:%M:%S %Y')   # update new_t when read a new input
                
                with open(file) as json_file:
                    json_data = json.load(json_file)
                # filename = file.split('/')[-1]
                # os.rename(file, testPath + '/.' + filename)   # hide file
                # os.remove(file)   # delete file
                if os.path.exists(os.path.join(backPath, filename)):
                    os.remove(os.path.join(backPath, filename))      # delete existing
                shutil.move(file, backPath)     # move file to backup path
                print(time.ctime(), 'New json input:', file)
                
                tensor = np.zeros(shape = (height, width, 1))
                for item in json_data['features']:
                    value = item['properties'][which_type]
                    tensor[height-1-item['properties']['index'][1]][item['properties']['index'][0]] = value*1e2 if value > 0 else 0 #value*1e4 if value > 0 else 0
                XS.append(tensor)
                
            if len(XS) >= timesteps and len(XS) != watcher:   # starts when seq len >=3
                testX = np.array(XS[-timesteps:])
                if modelName == 'STResNet':
                    testX = np.expand_dims(testX, axis=-2)
                    testX = np.squeeze(testX.swapaxes(-2, 0))
                testX = np.expand_dims(testX, axis=0)

                print('New test set dim:', testX.shape)
                y_hat = model.predict(testX)
                print('New prediction dim:', y_hat.shape)
                # print(y_hat)
                
                # output
                json_out(json_data, y_hat[0]/1e2, tempPath, filename, which_type)
                if os.path.exists(os.path.join(outputPath, filename)):
                    os.remove(os.path.join(outputPath, filename))      # delete existing pred
                shutil.move(tempPath + '/' + filename, outputPath)
                watcher = len(XS)
    return None

        
################# Parameter Settings #######################

timesteps = 3
channel = 1

LOSS = 'mse'
OPTIMIZER = 'adam'
LEARN = 1e-4
BATCHSIZE = 4
EPOCH = 100
SPLIT = 0.2

################## Main #######################

def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', default = 2, type = int,
                        choices = range(3), help = '0:test only / 1:train only / 2:train+test')
    parser.add_argument('--time_interval', default = 10, type = int, help = 'Time interval in minute')
    parser.add_argument('--which_type', default = 'normalize', type = str,
                        choices = ['count', 'stay', 'normalize'], help = "'count' or 'stay' or 'normalize'")
    parser.add_argument('--user_ID', type = str, help = 'Specify user ID')
    parser.add_argument('--user_name', type = str, help = 'Specify user name')
    #parser.add_argument('--model_name', type=str, default='STResNet', help='Specify model name')
    parser.add_argument('--GPU', type=str, help='Specify which GPU to run with (-1 for run on CPU)', default='-1')
    
    args = parser.parse_args()

    ######## GPU setting ########
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    if args.GPU == '-1':
        gpu_config = tf.ConfigProto(device_count={'GPU':0})
    else:
        gpu_config = tf.ConfigProto()
        #gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.visible_device_list = args.GPU
    set_session(tf.Session(config=gpu_config))
    
    ######## Directories ########
    trainPath = './' + args.user_ID + '/jsonfile'
    testPath = './' + args.user_ID + '/input'
    outputPath = './' + args.user_ID + '/output'
    
    tempPath = './' + args.user_ID + '/temp'
    #backPath = './' + args.user_ID + '/oldbackup'
    absbackPath = os.path.join(os.path.abspath(os.curdir), args.user_ID, 'oldbackup')
    
    if not os.path.exists(tempPath):
        os.makedirs(tempPath)
    if not os.path.exists(absbackPath):
        os.makedirs(absbackPath)

    # model name depends on which_type
    if args.which_type in ['count']:
        modelName = 'STResNet'
    elif args.which_type in ['stay', 'normalize']:
        modelName = 'ConvLSTM'

    # get train data
    trainX, trainY, height, width, max_ = getTrainXY(trainPath)
    
    # get model
    model = getModel(modelName, height, width)
    model.summary()
    model.compile(loss = LOSS, optimizer = OPTIMIZER)
    print(modelName + ' model compiled...')
    
    # train model
    if args.train_mode in [1, 2]:
        trainModel(model, modelName, trainX, trainY, tempPath)
    # test model
    if args.train_mode in [0, 2]:
        testModel(model, modelName, height, width, max_, args.which_type, args.user_name,
                  testPath, outputPath, tempPath, absbackPath)


if __name__ == '__main__':
    main()

