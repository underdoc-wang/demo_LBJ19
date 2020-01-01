import os
import json
import numpy as np
from glob import glob

timesteps = 3

def json_in(input_dir):
    input_seq = []
    max_ = 0

    for file in sorted(glob(os.path.join(input_dir, '*.json'))):
        with open(file) as json_file:
            json_data = json.load(json_file)
        
        nrow, ncol = json_data['index'][1], json_data['index'][0]
        # +time
        tensor = np.zeros(shape = (nrow, ncol, 1))
        for i, item in enumerate(json_data['values']):
            # filter each item
            item *= 1e4
            if item < 1:
                item = 0
            tensor[int(i/ncol)][ncol-i%ncol-1] = item if item >= 0 else 0     # filter out -1
            if item > max_:
                max_ = item
        
        input_seq.append(tensor)

    input_seq = np.array(input_seq)
    return input_seq, nrow, ncol, max_


def getTrainXY(train_dir):
    train_data, height, width, max_ = json_in(train_dir)
    print('Train set max value:', max_)
    train_data = train_data / max_      # normalize to 0-1

    XS, YS = [], []
    for i in range(len(train_data) - timesteps):
        x = train_data[i:i + timesteps]
        y = train_data[i + timesteps]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    print(XS.shape, YS.shape)

    return XS, YS, height, width, max_


def json_out(dic, y_hat, out_dir, out_name, which_type):
    for item in dic['features']:
        i = y_hat.shape[0] - 1 - item['properties']['index'][1]
        j = item['properties']['index'][0]
        # print(type(y_hat[i][j][0]))
        item['properties'][which_type] = float(y_hat[i][j][0])

    with open(out_dir + '/' + out_name, 'w') as outfile:
        json.dump(dic, outfile)
    return None


