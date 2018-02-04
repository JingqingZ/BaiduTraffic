
import numpy as np

import config
import dataloader

def mape(pred, target):
    return np.abs(pred - target) / target

def get_error(filename):

    print("Load Event")
    event_dict = dataloader.load_event_data()

    print("Loading Prediction")
    data = np.load(filename)
    test_pred = data["pred"]
    print("Prediction Loaded ", test_pred.shape)

    root_data, neighbour_data, pathlist = dataloader.load_data(5, 5)
    test_data = root_data[:, -config.valid_length:, :]

    print("Pred ", test_pred.shape)
    print("Test ", test_data.shape)

    import time

    path_time_loss = np.zeros((test_pred.shape[0], config.out_seq_length))
    path_event_loss = np.zeros((test_pred.shape[0], config.out_seq_length))

    steptime = time.time()
    # flagevent = 0
    for path in range(test_pred.shape[0]):
        # if flagevent > 10:
        #     break

        event_filter = dataloader.get_event_filter(event_dict[pathlist[path]])
        event_filter = np.array(event_filter)

        event_count = np.zeros(config.out_seq_length)

        if path % 1000 == 0:
            ntime = time.time()
            print(path, ntime - steptime)
            steptime = ntime

        for seqloc in range(test_pred.shape[1]):
            pred = test_pred[path, seqloc]
            real = test_data[path, seqloc + config.in_seq_length : seqloc + config.in_seq_length + config.out_seq_length]

            for tlen in range(config.out_seq_length):
                mapeloss = mape(pred[tlen, 0], real[tlen, 0])
                path_time_loss[path, tlen] += mapeloss

                if event_filter[seqloc + config.in_seq_length + tlen] > 0:
                    path_event_loss[path, tlen] += mapeloss
                    event_count[tlen] += 1

        path_time_loss[path] /= test_pred.shape[1]

        for tlen in range(config.out_seq_length):
            if event_count[tlen] > 0:
                path_event_loss[path, tlen] /= event_count[tlen]
        # if np.sum(event_count) > 0:
        #     print(path_event_loss[path])
            # flagevent += 1

    path_time_loss[path_time_loss == 0] = np.nan
    time_loss = np.nanmean(path_time_loss, axis=0)
    print("Time loss: ", time_loss)

    path_event_loss[path_event_loss == 0] = np.nan
    event_loss = np.nanmean(path_event_loss, axis=0)
    print("Event loss: ", event_loss)


if __name__ == "__main__":
    # get_error(config.result_path + "seq2seq_model/91_test_copy.npz")
    get_error(config.result_path + "spacial_model/101_test_copy.npz")
    get_error(config.result_path + "widedeep_model/101_test_copy.npz")
    pass
