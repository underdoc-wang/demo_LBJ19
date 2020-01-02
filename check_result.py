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
    test_data, height, width, max_ = json_in_test(back_dir, type)
    print('test shape', test_data.shape)
    print('test non-zero: \n', test_data[test_data > 0])

    pred_data, height, width, max_ = json_in_test(pred_dir, type)
    print('pred shape', pred_data.shape)
    print('pred non-zero: \n', pred_data[np.nonzero(pred_data)])

    for i in range(pred_data.shape[0]):
        plt.imshow(test_data[i+2,:,:,0])
        plt.savefig(images_dir + '/test/' + str(i)+'.png')

        plt.imshow(pred_data[i,:,:,0])
        plt.savefig(images_dir + '/pred/' + str(i)+'.png')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_type', default = 'count', type = str,
                        choices = ['count', 'stay'], help = "'count' or 'stay'")
    parser.add_argument('--user_ID', type = str, help = 'Specify user ID')
    parser.add_argument('--user_name', type = str, help = 'Specify user name')

    args = parser.parse_args()

    trainPath = './' + args.user_ID + '/jsonfile'
    outputPath = './' + args.user_ID + '/output'
    backPath = './' + args.user_ID + '/oldbackup'
    imgPath = './' + args.user_ID + '/images/stresnet'

    #check_train(trainPath)
    check_pred(outputPath, backPath, args.which_type, imgPath)



if __name__ == '__main__':
    main()