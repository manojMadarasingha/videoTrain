# this functino is to run the mlp on the 0.5s binned data created by Shashiks
# features: downloaded bytes amount is the feature to be updated.
import pandas as pd
import numpy as np
import os
import math
import argparse

from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, Flatten

from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# constant values
YT = 1
FB = 0

V360 = 1
VNormal = 0


# select the train and test ids according to this random state and the seen and unseen nature
def select_train_and_test_id(random_state, is_seen):
    # a set means id sets ranging from the 1-10,21-20,...
    if is_seen:
        test_ind = list(np.arange(1, 51))
        train_ind = list(np.arange(1, 51))
    else:
        num_of_ids_from_one_set = 3
        all_videos = np.arange(1, 51)
        test_ind = []

        # make sure that each video id is present in the test set at least 2 times
        for ind_set in range(5):

            for sub_id in range(num_of_ids_from_one_set):
                test_ind.append(ind_set * 10 + (sub_id + random_state))

        train_ind = list(set(list(all_videos)) - set(list(test_ind)))

    return train_ind, test_ind


# based on the train and test list get the original and synthesis dataset from the
# processed data
def truncate_train_list(temp_train_df, number_of_traces):
    temp_train_df = shuffle(temp_train_df, random_state=123)
    temp_train_df.reset_index(inplace=True, drop=True)
    len_temp_train_df = temp_train_df.shape[0]
    truncated_train_df = temp_train_df.loc[0:np.ceil(number_of_traces * len_temp_train_df) - 1, :]

    return truncated_train_df


def get_synth_df(synth_df, indices, num_of_synth_samples):
    temp_df_list = []

    for i in (indices):
        temp_df_list.append(
            synth_df.loc[i * num_of_synth_samples:i * num_of_synth_samples + num_of_synth_samples - 1, :])

    df = pd.concat(temp_df_list)

    # print(df.shape[0])
    return df


def get_processed_train_test_synth_orit(train_list, test_list, is_seen, number_of_traces, random_state, platform,path):
    main_path = path

    data_processed_types = ['original_data', 'synthesized_data']
    video_type = ['V360', 'VNormal']

    # split the dataset according to the seen condition. first 60% of traces are in the
    # train set and the remaining are in the test set
    if is_seen:

        data = [[], [], [], []]

        v_type_path_ori = main_path + '/original_data/' + platform + '/'
        v_type_path_synth = main_path + '/synthesized_data/' + platform + '/'

        ori_360 = pd.read_csv(v_type_path_ori + 'V360.csv')
        ori_normal = pd.read_csv(v_type_path_ori + '/' + 'VNormal.csv')
        synth_360 = pd.read_csv(v_type_path_synth + '/' + 'V360.csv')
        synth_normal = pd.read_csv(v_type_path_synth + '/' + 'VNormal.csv')

        data_types_ori = [ori_360, ori_normal]  # [ori_plus_synth_360,ori_plus_synth_normal]]
        data_types_synth = [synth_360, synth_normal]

        num_of_synth_samples = 20
        for v in range(len(data_types_ori)):
            df_ori = data_types_ori[v]
            df_synth = data_types_synth[v]

            train_ori = []
            test_ori = []
            train_synth = []
            test_synth = []

            for i in range(1, 51):
                #in the Facebook dataset video ID 32 is not appeared. Skip that ID
                if i == 32 and platform == 'Facebook':
                    continue

                # get the original dataframe
                t_df_ori_vid_selected = df_ori[df_ori['vid_id'] == i]
                ind_all = list(t_df_ori_vid_selected.index)

                # get the train df from ori
                len_vid_df = t_df_ori_vid_selected.shape[0]
                train_ind = math.ceil(len_vid_df * 0.6)

                # select train data from the original set
                train_df_rows_ori = t_df_ori_vid_selected.sample(train_ind, random_state=random_state)
                ind_train = list(train_df_rows_ori.index)

                # select train daat from the synth dataset
                train_df_rows_synth = get_synth_df(df_synth, ind_train, num_of_synth_samples)

                train_ori.append(train_df_rows_ori)
                train_synth.append(train_df_rows_synth)

                # select test dataset accordingly from both original and synthesized dataset
                if train_ind < len_vid_df:
                    ind_test = list(set(ind_all) - set(ind_train))

                    # get the synthesised dataset
                    test_df_rows_ori = t_df_ori_vid_selected.loc[ind_test, :]
                    test_df_rows_synth = get_synth_df(df_synth, ind_test, num_of_synth_samples)

                    test_ori.append(test_df_rows_ori)
                    test_synth.append(test_df_rows_synth)

            # format of the the data type arrangement in data[] variable
            # 360_ori, normal_ori, 360_synth, normal_synth

            final_ori_train = (pd.concat(train_ori))
            final_ori_test = (pd.concat(test_ori))
            final_synth_train = (pd.concat(train_synth))
            final_synth_test = (pd.concat(test_synth))

            data[v] = [final_ori_train, final_ori_test]
            data[v + 2] = [final_synth_train, final_synth_test]

    # select the data[] based on unseen conditoin.
    # simply split the dataset to train and test condition based on the video IDs given in train/test list
    else:
        data = []
        for p_type in data_processed_types:
            for v_type in video_type:
                df = pd.read_csv(main_path + '/' + p_type + '/' + platform + '/' + v_type + '.csv')
                # select the traces of the videos
                if p_type == 'original_data':
                    train_df_list = []
                    for i in train_list:
                        temp_train_df = df.loc[df['vid_id'] == i]
                        truncated_train_df = truncate_train_list(temp_train_df, number_of_traces)
                        train_df_list.append(truncated_train_df)
                    train_df = pd.concat(train_df_list)
                else:
                    train_df = df.loc[df['vid_id'].isin(train_list)]
                test_df = df.loc[df['vid_id'].isin(test_list)]

                data.append([train_df, test_df])

    return data


# run the ml model.
# num_of_trials: amount of synthesised data samples to be added
# data: 360_ori, normal_ori, 360_synth, normal_synth
def run_ml_cumulative_adding(num_of_samples, total_trials, data, is_synth_only, is_ori_plus_synth, is_mlp):
    ori_360 = data[0]
    ori_normal = data[1]
    # only select the train data from the synthesized data
    synth_360 = data[2][0]
    synth_normal = data[3][0]

    acc = []
    prec = []
    reca = []

    previous_synth_df = None
    remaining_synth_360_df = synth_360
    remaining_synth_normal_df = synth_normal

    print("")
    print("====== Start running " + str(total_trials) + " trials ======")

    if is_ori_plus_synth:
        running_trials = total_trials + 1
    else:
        running_trials = total_trials

    for trial in range(0, running_trials):

        # ensure that we cumulatively add synthesized data sampling traces without replacement from the
        # train set of synthesized dat. We assess the model only on the test data of original data set
        if (is_ori_plus_synth and trial > 0) or is_synth_only:

            sampled_synth_360 = remaining_synth_360_df.sample(num_of_samples,
                                                              axis=0,
                                                              random_state=(trial + 1) * 10,
                                                              replace=False)
            sampled_synth_normal = remaining_synth_normal_df.sample(num_of_samples, axis=0,
                                                                    random_state=(trial + 1) * 10,
                                                                    replace=False)

            remaining_synth_360_df = remaining_synth_360_df.merge(sampled_synth_360, indicator=True, how='left').loc[
                lambda x: x['_merge'] != 'both']
            remaining_synth_normal_df = \
                remaining_synth_normal_df.merge(sampled_synth_normal, indicator=True, how='left').loc[
                    lambda x: x['_merge'] != 'both']

            remaining_synth_360_df.drop(['_merge'], inplace=True, axis=1)
            remaining_synth_normal_df.drop(['_merge'], inplace=True, axis=1)

            current_synth_df = pd.concat([sampled_synth_360, sampled_synth_normal], axis=0)

            if trial > 0:
                current_synth_df = pd.concat([current_synth_df, previous_synth_df], axis=0)

            previous_synth_df = current_synth_df.copy()

        # if it is original + synthsized data, combine the sampled synthesized dataset with the original dataset
        if is_ori_plus_synth:
            if trial == 0:
                train_data = pd.concat([ori_360[0], ori_normal[0]], axis=0)
            else:
                train_data = pd.concat([ori_360[0], ori_normal[0], current_synth_df], axis=0)

        # if it is only synthesized data, add only the synthesized dataset
        elif is_synth_only:
            train_data = current_synth_df

        else:
            train_data = pd.concat([ori_360[0], ori_normal[0]], axis=0)

        test_data = pd.concat([ori_360[1], ori_normal[1]])

        # drop the video id
        train_data.drop('vid_id', axis=1, inplace=True)
        test_data.drop('vid_id', axis=1, inplace=True)

        train_data = shuffle(train_data, random_state=123)

        # create X and Y dataset
        target = 'vid_type'
        predictors = [x for x in train_data.columns if x not in [target]]

        train_data.to_csv()
        x_train = train_data[predictors].values
        x_test = test_data[predictors].values
        y_train = train_data[target].values
        y_test = test_data[target].values

        # normaize the features set
        len_train = x_train.shape[0]
        combined_dataset_arr = np.concatenate([x_train, x_test])

        # combined_dataset_arr = combined_data.values
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        combined_dataset_arr = scaler.transform(combined_dataset_arr)

        x_train = combined_dataset_arr[:len_train, :]
        x_test = combined_dataset_arr[len_train:, :]

        # run mlp model upon selection
        if is_mlp:
            input_dim = x_train.shape[1]
            model_mlp = Sequential()
            model_mlp.add(Dense(300, input_dim=input_dim, activation='relu'))
            model_mlp.add(BatchNormalization())
            model_mlp.add(Dense(100, activation='relu'))
            model_mlp.add(Dropout(0.5))
            model_mlp.add(Dense(10, activation='relu'))
            model_mlp.add(Dense(1, activation='sigmoid'))

        # run cnn model upon selection
        else:
            dim_train = x_train.shape
            dim_test = x_test.shape
            x_train = x_train.reshape([dim_train[0], dim_train[1], 1])
            x_test = x_test.reshape([dim_test[0], dim_test[1], 1])
            # y_train = to_categorical(y_train)
            # y_test = to_categorical(y_test)

            input_dim = (x_train.shape[1], x_train.shape[2])

            n_outputs = 1
            model_cnn = Sequential()
            model_cnn.add(Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=input_dim))
            model_cnn.add(BatchNormalization())
            model_cnn.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
            model_cnn.add(Dropout(0.5))
            model_cnn.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
            model_cnn.add(MaxPooling1D(pool_size=4))
            model_cnn.add(Conv1D(filters=16, kernel_size=8, activation='relu'))
            model_cnn.add(Dropout(0.5))
            model_cnn.add(Flatten())
            model_cnn.add(Dense(32, activation='relu'))
            model_cnn.add(Dense(n_outputs, activation='sigmoid'))

        if is_mlp:
            model = model_mlp
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model = model_cnn
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=True)

        y_pred = model.predict(x_test).flatten()
        y_pred = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_pred, y_test)
        precsion = average_precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')

        acc.append(accuracy)
        prec.append(precsion)
        reca.append(recall)

        print('       Accuracy is:', accuracy * 100)

    return (acc), (prec), (reca)


# read the process data
# generate random sets
# feed them for the MLP network

def read_processed_data_and_process(platform, total_trials, is_ori_plus_synth, is_synth_only, is_seen, is_mlp, path):
    # by default set these values for the number of different train/test splits.
    # In unseen condition random sets = 9 ensures traces from each video ID is fallen
    # at least 2 times in the test set and remaining splits in the train set. In
    # seen condition having only 9 train/test splits ensure each trace from each video appears
    # in the test set at least 2 times.
    # to get the final results average the performance over all the random sets3

    if is_seen:
        num_of_random_sets = 4
    else:
        num_of_random_sets = 9

    total_trials = total_trials
    num_of_samples = 100
    num_of_original_videos = 1

    for r in range(1, num_of_random_sets):  # r should start with 1

        train_list, test_list = select_train_and_test_id(random_state=r, is_seen=is_seen)

        print(train_list)

        # based on the sen an unseen condition split dataset into train and test splits both for
        # original and synthesized dataset.
        data = get_processed_train_test_synth_orit(train_list, test_list,
                                                   is_seen=is_seen,
                                                   number_of_traces=num_of_original_videos,
                                                   random_state=r,
                                                   platform=platform,
                                                   path = path)

        acc, prec, rec = run_ml_cumulative_adding(num_of_samples=num_of_samples,
                                                  total_trials=total_trials,
                                                  data=data,
                                                  is_synth_only=is_synth_only,
                                                  is_ori_plus_synth=is_ori_plus_synth,
                                                  is_mlp=False)

        performance = np.asarray([acc, prec, rec])
        performance = np.swapaxes((performance), 0, 1)

        if is_synth_only:
            trial = np.arange(num_of_samples * 2, num_of_samples * 2 * total_trials + 1,
                              num_of_samples * 2).reshape([-1, 1])
        if is_ori_plus_synth:
            trial = np.arange(0, num_of_samples * 2 * total_trials + 1, num_of_samples * 2).reshape(
                [-1, 1])

        data = np.concatenate((trial, performance), axis=1)
        cols = ['Trials', 'Acc', 'Prec', 'Reca']
        df = pd.DataFrame(columns=cols,
                          data=data)

        if is_seen:
            seen_cond = 'seen'

        else:
            seen_cond = 'unseen'

        if is_mlp:
            dnn = "MLP"
        else:
            dnn = "CNN"

        path = path + '/results/' + platform + '/' + dnn + "/" + seen_cond

        if not os.path.exists(path):
            os.makedirs(path)

        if is_synth_only:
            path += '/only_synth'
        elif is_ori_plus_synth:
            path += '/ori_plus_synth'
        else:
            path += '/only_ori'

        path += '_set_' + str(r) + '.csv'
        df.to_csv(path)

    return


def get_cmd_arg():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--t_type',
                        type=str,
                        default='YT',
                        help='Select one of the traffic type as a String variable: ' + '"' + 'YT' + '",' + '"' + 'FB' + '"')

    parser.add_argument('--data_addition_rounds',
                        type=int,
                        default=10,
                        help="num of rounds (int) that the synthesized data been added. At each round by default 100 new traces are added from each category (360/Normal)."
                             "Having at maximum 10 rounds (default set) is enough to acheive the required results")

    parser.add_argument('--only_synth_traces',
                        default=False,
                        action='store_true',
                        help='Consider only synthesised traces for the analysis')

    parser.add_argument('--is_unseen',
                        default=False,
                        action='store_true',
                        help='Unseen scenario in which non of the traces in train video IDs appear in the test set')

    parser.add_argument('--mlp',
                        default=False,
                        action='store_true',
                        help="Run the MLP model. Default model: CNN")

    parser.add_argument('--path',
                        type=str,
                        # set default path to the current working directory.
                        default=os.path.abspath(os.getcwd()),
                        help="Set the working dir path. Default: current working directory")

    args = parser.parse_args()

    # vaidate each parameter and pass to the main function
    # platform
    pltform_str = args.t_type
    if pltform_str == 'YT':
        pltform = 'YouTube'
    elif pltform_str == 'FB':
        pltform = 'Facebook'
    else:
        pltform = None
        print('Enter valid Platform name')
        exit(0)

    # total addition rounds
    total_trials = args.data_addition_rounds

    # synth only or original + synth traces
    is_ori_plus_synth = not args.only_synth_traces
    is_synth_only = args.only_synth_traces

    # seen unsen condition
    is_seen = not args.is_unseen

    # is dnn condition
    is_mlp = args.mlp

    # patht to the data
    path = args.path

    print('platform :' + pltform)
    print('total_trials :' + str(total_trials))
    print('is_ori_plus_synth :' + str(is_ori_plus_synth))
    print('is_synth_only :' + str(is_synth_only))
    print('is_seen :' + str(is_seen))
    print('is_mlp :' + str(is_mlp))
    print('path :' + path)

    return pltform, total_trials, is_ori_plus_synth, is_synth_only, is_seen, is_mlp, path


def main():
    pltform, total_trials, is_ori_plus_synth, is_synth_only, is_seen, is_mlp, path = get_cmd_arg()

    read_processed_data_and_process(pltform, total_trials, is_ori_plus_synth, is_synth_only, is_seen, is_mlp, path)

    return


if __name__ == main():
    main()
