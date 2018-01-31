
import pickle
import queue
import numpy as np
import progressbar

import config

def find_neighbours(predecessor=5, successors=5):

    event_filter_file = open(config.data_path + "event_filter.txt", "r")
    event_filter_flag = eval(event_filter_file.readline())
    event_filter_file.close()

    event_set_file = open(config.data_path + "event_link_set_beijing_1km", "r")
    event_set = event_set_file.readlines()
    event_set_file.close()

    event_link_file = open(config.result_path + "event_link_set_beijing_link_1km.txt", "r")
    event_link = event_link_file.readlines()
    event_link_file.close()

    event_pagerank_file = open(config.result_path + "pagerank_1km.txt", "r")
    event_pagerank = event_pagerank_file.readlines()
    event_pagerank_file.close()

    # traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_mv_avg_15min_completion.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min.pkl", "rb")
    traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    outfile = open(config.result_path + "neighbours_1km.txt", "w")

    assert len(event_filter_flag) == len(event_set)
    assert len(event_filter_flag) == len(event_link)
    assert len(event_filter_flag) == len(event_pagerank)

    save_dict = dict()

    bar = progressbar.ProgressBar(max_value=1151)
    for iter, line in enumerate(event_set):
        bar.update(iter)

        if event_filter_flag[iter] == 0:
            continue
        nodes = line.replace("\n", "").split("\t")

        def get_dict(links):
            ldict = dict()
            llist = eval(links)
            for l in llist:
                if l[2] != 1:
                    continue
                if l[0] not in ldict:
                    ldict[l[0]] = {"prev": list(), "next": list()}
                if l[1] not in ldict:
                    ldict[l[1]] = {"prev": list(), "next": list()}
                ldict[l[0]]["next"].append(l[1])
                ldict[l[1]]["prev"].append(l[0])
            return ldict

        link_dict = get_dict(event_link[iter])

        for node in nodes:
            if node not in traffic_data:
                continue

            def get_neighbours(root, num, direction):
                assert direction == "prev" or direction == "next"
                qnode = queue.Queue()
                qnode.put(root)
                reslist = list()
                while len(reslist) < num and not qnode.empty():
                    cur = qnode.get()
                    if cur not in traffic_data:
                        continue
                    if cur in link_dict:
                        for n in link_dict[cur][direction]:
                            qnode.put(n)
                    if cur != root:
                        reslist.append(cur)
                return reslist

            prevlist = get_neighbours(node, predecessor, "prev")
            nextlist = get_neighbours(node, successors, "next")

            if len(prevlist) < predecessor or len(nextlist) < successors:
                continue

            fulllist = [node]
            fulllist += prevlist
            fulllist += nextlist

            key = ""
            for ele in fulllist:
                key += ele + "_"
            if key in save_dict:
                continue
            save_dict[key] = 1

            # print(fulllist)

            outfile.write(str(fulllist))
            outfile.write("\n")
    outfile.close()

def load_data(predecessor=5, successors=5):
    print("Loading data...")
    # traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt_0.05.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_mv_avg_15min_completion.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min.pkl", "rb")
    traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    neighbour_file = open(config.result_path + "neighbours_1km.txt", "r")
    neighbour = neighbour_file.readlines()

    rootdata = list()
    neigdata = list()

    rootpathlist = list()
    for line in neighbour:
        group = eval(line)
        assert len(group) == predecessor + successors + 1

        root_traffic = traffic_data[group[0]]
        root_traffic = np.expand_dims(root_traffic, axis=-1)

        prevlist = group[1 : 1 + predecessor]
        nextlist = group[-successors : ]
        neighbourlist = list()

        for prev in reversed(prevlist):
            neighbourlist.append(traffic_data[prev])
        for next in nextlist:
            neighbourlist.append(traffic_data[next])

        neighbourlist = np.array(neighbourlist)
        neighbourlist = np.swapaxes(neighbourlist, 0, 1)

        rootdata.append(root_traffic)
        neigdata.append(neighbourlist)

        rootpathlist.append(group[0])

    rootdata = np.stack(rootdata)
    neigdata = np.stack(neigdata)

    assert rootdata.shape[:-1] == neigdata.shape[:-1]

    lowbound = 5
    assert lowbound > 0
    assert lowbound < np.percentile(rootdata, 2)
    rootdata[rootdata < lowbound] = lowbound
    neigdata[neigdata < lowbound] = lowbound

    print("Data Loaded: x_root %s, x_neighbour: %s" % (rootdata.shape, neigdata.shape))

    return rootdata, neigdata, rootpathlist

def get_minibatch(root_data, neighbour_data, order, num_seq):
    minibatch_x_root = list()
    minibatch_x_neighbour = list()
    minibatch_y_root = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])
        minibatch_x_neighbour.append(neighbour_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_x_neighbour = np.stack(minibatch_x_neighbour)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_x_neighbour, minibatch_decode_seq, minibatch_target_seq

def load_data_all():
    print("Loading data...")
    # traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt_0.05.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_mv_avg_15min_completion.pkl", "rb")
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min.pkl", "rb")
    traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    alldata = list()
    nodelist = list()
    for node in traffic_data:
        alldata.append(traffic_data[node])
        nodelist.append(node)
    alldata = np.stack(alldata)
    alldata = np.expand_dims(alldata, axis=-1)

    # print(np.isfinite(alldata))
    # print(np.sum(alldata))
    lowbound = 5
    assert lowbound > 0
    assert lowbound < np.percentile(alldata, 2)
    alldata[alldata < lowbound] = lowbound

    print("Data Loaded: all ", alldata.shape)
    return alldata, nodelist

def get_minibatch_all(root_data, order, num_seq):
    minibatch_x_root = list()
    minibatch_y_root = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq

def get_minibatch_features(root_data, features_info, features_time, order, num_seq):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_features = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])

        f = np.zeros([config.out_seq_length + 1, config.dim_features])
        for fi in range(config.out_seq_length):
            f[fi, :config.dim_features_info] = features_info[seq_id, :]
            f[fi, -config.dim_features_time:] = features_time[seq_loc + config.in_seq_length + fi, :]
        minibatch_features.append(f)

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_features = np.stack(minibatch_features)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_features, minibatch_decode_seq, minibatch_target_seq

def get_minibatch_features_4_test(root_data, features_info, features_time, path, cstep):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_features = list()

    for o in range(config.batch_size):
        baseloc = o + cstep * config.batch_size
        minibatch_x_root.append(root_data[path, baseloc : baseloc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])

        f = np.zeros([config.out_seq_length + 1, config.dim_features])
        for fi in range(config.out_seq_length):
            f[fi, :config.dim_features_info] = features_info[path, :]
            f[fi, -config.dim_features_time:] = features_time[-config.valid_length + baseloc + config.in_seq_length + fi, :]
        minibatch_features.append(f)

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_features = np.stack(minibatch_features)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_features, minibatch_decode_seq, minibatch_target_seq

def get_minibatch_4_test(root_data, path, cstep):
    minibatch_x_root = list()
    minibatch_y_root = list()

    for o in range(config.batch_size):
        baseloc = o + cstep * config.batch_size
        minibatch_x_root.append(root_data[path, baseloc : baseloc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq

def get_minibatch_4_test_neighbour(root_data, neighbour_data, path, cstep):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_x_neighbour = list()

    for o in range(config.batch_size):
        baseloc = o + cstep * config.batch_size
        minibatch_x_root.append(root_data[path, baseloc : baseloc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])
        minibatch_x_neighbour.append(neighbour_data[path, baseloc : baseloc + config.in_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_x_neighbour = np.stack(minibatch_x_neighbour)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_x_neighbour, minibatch_decode_seq, minibatch_target_seq

def load_features(pathlist=None):
    print("Loading Features ...")
    coarse_file = open(config.data_path + "wide_features/event_link_set_all_poi_type_feature_coarse_beijing_1km.pkl", "rb")
    fine_file = open(config.data_path + "wide_features/event_link_set_all_poi_type_feature_fine_beijing_1km.pkl", "rb")
    info_file = open(config.data_path + "wide_features/event_link_set_all_beijing_1km_link_info_feature.pkl", "rb")
    time_file = open(config.data_path + "wide_features/time_feature_15min.pkl", "rb")

    linklist_coarse, features_coarse = pickle.load(coarse_file, encoding='latin1')
    linklist_fine, features_fine = pickle.load(fine_file, encoding='latin1')
    linklist_info, features_info = pickle.load(info_file, encoding='latin1')
    features_time = pickle.load(time_file, encoding='latin1')

    coarse_file.close()
    fine_file.close()
    info_file.close()
    time_file.close()

    assert linklist_coarse == linklist_fine
    assert linklist_coarse == linklist_info
    linklist = linklist_coarse

    if pathlist is not None:
        new_features_info = np.zeros((len(pathlist), features_info.shape[1]))

        for pid, path in enumerate(pathlist):
            linkidx = linklist.index(path)
            new_features_info[pid, :] = features_info[linkidx, :]

        features_info = new_features_info
        linklist = pathlist

    print("Features Loaded. Info %s, Time %s" % (features_info.shape, features_time.shape))

    return features_info, features_time, linklist


if __name__ == "__main__":
    # find_neighbours(5, 5)
    # r, n, p = load_data(5, 5)
    # get_minibatch(r, n, order=[0,1], num_seq=r.shape[1] - (config.in_seq_length + config.out_seq_length) + 1)
    '''
    r = load_data_all()
    import time
    st = time.time()
    get_minibatch_all(r, order=list(range(config.batch_size)), num_seq=r.shape[1] - (config.in_seq_length + config.out_seq_length) + 1)
    print(time.time() - st)
    '''
    fi, ft, fp = load_features(["1525826704", "1561981475"])
    print(fi)
    print(ft)
    print(fp)
    pass

