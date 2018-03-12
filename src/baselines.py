import numpy as np
import random
import sklearn
import time
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA

import utils
import config
import dataloader

def get_xy(data, testlen=1, filter_num=30000, event=False):
    x = list()
    y = list()
    num_seq = data.shape[1] - config.in_seq_length - config.out_seq_length

    allorder = list(range(data.shape[0] * num_seq))
    random.shuffle(allorder)
    if event:
        pathlist = dataloader.get_pathlist()
        event_data = dataloader.load_event_data()
        event_filter_allpath = dataloader.get_event_filter_allpath(event_data, pathlist)
        filterorder = dataloader.get_event_orders(event_filter_allpath, allorder, num_seq, tsteps=filter_num // config.batch_size)
    else:
        filterorder = allorder[:filter_num]

    # for path in range(data.shape[0]):
    #     for xid in range(data.shape[1] - config.in_seq_length - config.out_seq_length):
    for fid in filterorder:
        path = fid // num_seq
        xid = fid % num_seq
        x.append(data[path, xid : xid + config.in_seq_length])
        if testlen == 1:
            y.append(data[path, xid + config.in_seq_length])
        else:
            y.append(data[path, xid + config.in_seq_length : xid + config.in_seq_length + testlen])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def test_baseline(model, filter_num, testfilter_num):
    data, pathlist = dataloader.load_data_noneighbour(5, 5)

    traindata = data[:, :-config.valid_length // 2, 0]
    testdata = data[:, -config.valid_length:, 0]

    print("Get XY ...")
    x, y = get_xy(traindata, 1, filter_num)
    print("Get Train XY %s %s " % (x.shape, y.shape))
    tx, ty = get_xy(testdata, config.out_seq_length, testfilter_num)
    print("Get Test XY %s %s " % (tx.shape, ty.shape))
    ex, ey = get_xy(testdata, config.out_seq_length, testfilter_num, event=True)
    print("Get Etest XY %s %s " % (ex.shape, ey.shape))

    # x, y = sklearn.utils.shuffle(x, y)
    # x = x[:30000, :]
    # y = y[:30000]

    # tx, ty = sklearn.utils.shuffle(tx, ty)
    # tx = tx[:30000, :]
    # ty = ty[:30000]

    print("Train")
    stime = time.time()
    model.fit(x, y)
    etime = time.time()
    print("Train %d" % (etime - stime))

    def test(testx, testy):
        print("Test")
        stime = time.time()
        predlist = list()
        for pid in range(config.out_seq_length):
            print("Test %d %d" % (pid, time.time() - stime))
            pred = model.predict(testx)
            predlist.append(pred)
            testx[:, :-1] = testx[:, 1:]
            testx[:, -1] = pred
        etime = time.time()
        print("Test %d" % (etime - stime))

        predlist = np.stack(predlist, axis=-1)
        mapeloss = utils.mape(predlist, testy)
        tloss = np.mean(mapeloss, axis=0)
        print("Test ", tloss)

    test(tx, ty)
    test(ex, ey)
    # print(sumtloss / data.shape[0])

def test_arima(testfilter_num):
    data, pathlist = dataloader.load_data_noneighbour(5, 5)

    # traindata = data[:, :-config.valid_length // 2, 0]
    testdata = data[:, -config.valid_length:, 0]

    print("Get XY ...")
    # x, y = get_xy(traindata, 1, filter_num)
    # print("Get Train XY %s %s " % (x.shape, y.shape))
    tx, ty = get_xy(testdata, config.out_seq_length, testfilter_num)
    print("Get Test XY %s %s " % (tx.shape, ty.shape))
    ex, ey = get_xy(testdata, config.out_seq_length, testfilter_num, event=True)
    print("Get Etest XY %s %s " % (ex.shape, ey.shape))

    # x, y = sklearn.utils.shuffle(x, y)
    # x = x[:30000, :]
    # y = y[:30000]

    # tx, ty = sklearn.utils.shuffle(tx, ty)
    # tx = tx[:30000, :]
    # ty = ty[:30000]

    def test(testx, testy):
        print("Test")
        stime = time.time()
        predlist = list()
        for idx in range(testx.shape[0]):
            model = ARIMA(testx[0], order=(1, 1, 0))
            model_fit = model.fit(disp=0)
            pred, stderr, conf_int = model_fit.forecast(config.out_seq_length)
            predlist.append(pred)
            if idx % 20 == 0:
                print("Test %d %d" % (idx, time.time() - stime), utils.mape(pred, testy[idx]))

        etime = time.time()
        print("Test %d" % (etime - stime))
        predlist = np.stack(predlist, axis=0)
        mapeloss = utils.mape(predlist, testy)
        tloss = np.mean(mapeloss, axis=0)
        print("Test ", tloss)
        return tloss

    timeloss = test(tx, ty)
    eventloss = test(ex, ey)
    print(timeloss, np.mean(timeloss))
    print(eventloss, np.mean(eventloss))
    # print(sumtloss / data.shape[0])


if __name__ == "__main__":
    # test_baseline(model=SVR(kernel="rbf", epsilon=0.2), filter_num=1000, testfilter_num=1000000)
    # test_baseline(model=SVR(kernel="linear", epsilon=0.1), filter_num=1000, testfilter_num=1000000)
    # test_baseline(model=RandomForestRegressor(max_depth=10), filter_num=10000, testfilter_num=1000000)
    # test_baseline(model=GaussianProcessRegressor(), filter_num=1000)
    test_arima(testfilter_num=1000)
    pass
