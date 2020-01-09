import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_utils import json_in_train, json_in_test


def check_train(train_dir):
    train_data, height, width, max_ = json_in_train(train_dir)

    plt.hist(train_data.flatten())
    plt.savefig('./train_distribution.png')

    return None

def check_pred(pred_dir, back_dir, type, images_dir):
    test_data, H, W = json_in_test(back_dir, type)
    print('test shape', test_data.shape)
    print('test non-zero: \n', test_data[test_data > 0].shape[0])

    pred_data, H, W = json_in_test(pred_dir, type)
    print('pred shape', pred_data.shape)
    print('pred non-zero: \n', pred_data[np.nonzero(pred_data)].shape[0])

    # print images
    step = 0
    # ground truth
    for i in range(4):
        plt.imshow(test_data[step+i,:,:,0])
        plt.savefig(os.path.join(images_dir, str(i)+'.png'))

    # predicted value
    plt.imshow(pred_data[step,:,:,0])
    plt.savefig(os.path.join(images_dir, str(i)+'_pred.png'))

    return None

def evaluate_result(pred_dir, test_dir, type):
    pred_data, H_pred, W_pred = json_in_test(pred_dir, type)
    test_data, H_test, W_test = json_in_test(test_dir, type)
    assert H_pred == H_test, 'pred-test not in same dimension: H'
    assert W_pred == W_test, 'pred-test not in same dimension: W'

    # evaluate
    y_pred = pred_data[:-1]
    y_true = test_data[3:]
    assert y_pred.shape == y_true.shape, 'pred-true not in same dimension'
    print(y_pred.shape)

    mse = ((y_pred - y_true)**2).mean(axis=None)
    print('Total MSE:', round(mse, 4))


    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_type', default = 'normalize', type = str,
                        choices = ['count', 'stay', 'normalize'], help = "'count' or 'stay' or 'normalize'")
    parser.add_argument('--user_ID', type = str, help = 'Specify user ID')
    parser.add_argument('--user_name', type = str, help = 'Specify user name')
    #parser.add_argument('--model_name', type=str, default='STResNet', help='Specify model name')

    args = parser.parse_args()

    # model name depends on which_type
    if args.which_type in ['count']:
        modelName = 'STResNet'
    elif args.which_type in ['stay', 'normalize']:
        modelName = 'ConvLSTM'

    trainPath = './' + args.user_ID + '/jsonfile'
    outputPath = './' + args.user_ID + '/output'
    backPath = './' + args.user_ID + '/oldbackup'
    imgPath = './' + args.user_ID + '/images/' + modelName

    #check_train(trainPath)
    check_pred(outputPath, backPath, args.which_type, imgPath)
    evaluate_result(outputPath, backPath, args.which_type)


if __name__ == '__main__':
    main()