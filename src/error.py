
import numpy as np

import config
import dataloader

def mape(pred, target):
    return np.abs(pred - target) / target

def get_error(filename, model_name):

    print("Load Event")
    event_dict = dataloader.load_event_data()

    print("Loading Prediction %s" % filename)
    data = np.load(filename)
    test_pred = data["pred"]
    print("Prediction Loaded ", test_pred.shape)

    # root_data, neighbour_data, pathlist = dataloader.load_data(5, 5)
    root_data, pathlist = dataloader.load_data_noneighbour(5, 5)
    test_data = root_data[:, -config.valid_length:, :]

    print("Pred ", test_pred.shape)
    print("Test ", test_data.shape)

    import time

    path_time_loss = np.zeros((test_pred.shape[0], config.out_seq_length))
    path_event_loss = np.zeros((test_pred.shape[0], config.out_seq_length))

    path_period_loss = np.zeros((test_pred.shape[0], test_pred.shape[1]))

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

                if tlen == config.out_seq_length - 1:
                    path_period_loss[path, seqloc] = mapeloss

        path_time_loss[path] /= test_pred.shape[1]

        for tlen in range(config.out_seq_length):
            if event_count[tlen] > 0:
                path_event_loss[path, tlen] /= event_count[tlen]
        # if np.sum(event_count) > 0:
        #     print(path_event_loss[path])
            # flagevent += 1
    np.savez(config.result_path + "%s_path_period_loss.npz" % model_name, error=path_period_loss)
    print("Path period loss saved")

    path_time_loss[path_time_loss == 0] = np.nan
    time_loss = np.nanmean(path_time_loss, axis=0)
    print("Time loss: ", time_loss)

    path_event_loss[path_event_loss == 0] = np.nan
    event_loss = np.nanmean(path_event_loss, axis=0)
    print("Event loss: ", event_loss)

def sort_path_error(filename):
    data = np.load(filename)
    data = data["error"]
    pathlist = dataloader.get_pathlist()

    pathmean = np.mean(data, axis=1)
    print(pathmean.shape)


def get_event_loss(filename, filename_event):

    print("Load event")
    eventfile = open(filename_event, "r")
    eventdict = dict()
    for line in eventfile:
        content = line.replace("\n", "").split(":")
        eventdict[content[0]] = eval(content[1])


    print("Loading Prediction %s" % filename)
    data = np.load(filename)
    test_pred = data["pred"]
    print("Prediction Loaded ", test_pred.shape)

    root_data, pathlist = dataloader.load_data_noneighbour(5, 5)
    # test_data = root_data[:, -config.valid_length:, :]
    print("Test data ", root_data.shape)
    print("Pathlist %d" % len(pathlist))

    path_event_loss = np.zeros((test_pred.shape[0], config.out_seq_length))

    testedpath = dict()

    for pid in range(test_pred.shape[0]):
        path = pathlist[pid]
        if path not in eventdict or path in testedpath:
            continue

        testedpath[path] = 0

        eventlist = eventdict[path]

        if len(eventlist) == 0:
            continue

        numiter = 0

        last = -1

        # print(eventlist)
        for t in eventlist:
            if t < last + 50:
                last = t
                continue
            last = t
            # print(last)

            maxseqloc = t - (config.full_length - config.valid_length + config.in_seq_length)
            if maxseqloc >= test_pred.shape[1] or maxseqloc < 0:
                continue
            numiter += 1

            real =  root_data[pid, t]
            for ol in range(test_pred.shape[2]):
                seqloc = t - (config.full_length - config.valid_length + config.in_seq_length + ol)
                pred = test_pred[pid, seqloc, ol]
                mapeloss = mape(pred, real)
                path_event_loss[pid, ol] += mapeloss

        path_event_loss[pid] /= numiter

        if pid % 1000 == 0:
            tmp = path_event_loss.copy()
            tmp[tmp == 0] = np.nan
            event_loss = np.nanmean(tmp, axis=0)
            print("Event loss %d: " % pid, event_loss)

    path_event_loss[path_event_loss == 0] = np.nan
    event_loss = np.nanmean(path_event_loss, axis=0)
    print("Event loss: ", event_loss)



if __name__ == "__main__":
    # get_error(config.result_path + "seq2seq_model/91_test_copy.npz", "seq2seq_model_91")
    # get_error(config.result_path + "spacial_model/101_test_copy.npz", "spatial_model_101")
    # get_error(config.result_path + "widedeep_model/101_test_copy.npz", "widedeep_model_101")
    # get_error(config.result_path + "query_comb_model/101_test_notsample_50_copy.npz", "query_comb_model")
    # get_error(config.result_path + "query_comb_model/141_test_sample_50_copy.npz", "query_comb_model")
    # get_error(config.result_path + "query_comb_model_150/101_test_filt_copy.npz", "query_comb_model")
    # get_error(config.result_path + "query_comb_model_150/21_test_filt_update_copy.npz", "query_comb_model_21")
    # get_error(config.result_path + "all_comb_model_150_128_128/131_test_spatial_wide_copy.npz", "spatial_wide_comb_131")
    # get_error(config.result_path + "all_comb_model_150_128_128/141_test_hybrid_copy.npz", "hybrid_comb_141")
    get_error(config.result_path + "all_comb_model_150_128_128/141_test_res_copy.npz", "hybrid_comb_res_141")

    # sort_path_error(config.result_path + "query_comb_model_21_path_period_loss.npz")

    '''
    get_event_loss(
        # config.result_path + "seq2seq_model/91_test_copy.npz",
        config.result_path + "query_comb_model_150/21_test_filt_update_copy.npz",
        # config.result_path + "spacial_model/101_test_copy.npz",
        # config.result_path + "widedeep_model/101_test_copy.npz",
        # config.result_path + "all_comb_model_150_128_128/141_test_hybrid_copy.npz",
        # config.result_path + "all_comb_model_150_128_128/131_test_spatial_wide_copy.npz",

        config.data_path + "selected_link_time/selected_link_time_avg_day_std_2.txt"
        # config.data_path + "selected_link_time/selected_link_time_avg_day_std_3.txt"
        # config.data_path + "selected_link_time/selected_link_time_std_2.txt"
    )
    '''
    pass
