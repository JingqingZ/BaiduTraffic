
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

def get_pathlist():

    pathlist = list()
    neighbour_file = open(config.result_path + "neighbours_1km.txt", "r")
    neighbour = neighbour_file.readlines()

    for line in neighbour:
        group = eval(line)
        pathlist.append(group[0])

    return pathlist

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

def get_minibatch_all_query(root_data, query_data, pathlist, order, num_seq):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_query_x = list()
    minibatch_query_y = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])
        minibatch_query_x.append(query_data[pathlist[seq_id]][seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_query_y.append(query_data[pathlist[seq_id]][seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_query_x = np.stack(minibatch_query_x)
    minibatch_query_y = np.stack(minibatch_query_y)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_decode_seq_query = np.zeros((minibatch_query_y.shape[0], minibatch_query_y.shape[1] + 1, minibatch_query_y.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root
    minibatch_decode_seq_query[:, 1: ,:] = minibatch_query_y

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id
    minibatch_decode_seq_query[:, 0, :] = config.start_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq, minibatch_query_x, minibatch_decode_seq_query

def get_minibatch_all_comb(root_data, neighbour_data, features_info, features_time, query_data, pathlist, order, num_seq):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_query_x = list()
    minibatch_query_y = list()
    minibatch_x_neighbour = list()
    minibatch_features = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])
        minibatch_query_x.append(query_data[pathlist[seq_id]][seq_loc : seq_loc + config.in_seq_length, :])
        minibatch_query_y.append(query_data[pathlist[seq_id]][seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length, :])
        minibatch_x_neighbour.append(neighbour_data[seq_id, seq_loc : seq_loc + config.in_seq_length, :])

        f = np.zeros([config.out_seq_length + 1, config.dim_features])
        for fi in range(config.out_seq_length):
            f[fi, :config.dim_features_info] = features_info[seq_id, :]
            f[fi, -config.dim_features_time:] = features_time[seq_loc + config.in_seq_length + fi, :]
        minibatch_features.append(f)


    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_query_x = np.stack(minibatch_query_x)
    minibatch_query_y = np.stack(minibatch_query_y)
    minibatch_x_neighbour = np.stack(minibatch_x_neighbour)
    minibatch_features = np.stack(minibatch_features)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_decode_seq_query = np.zeros((minibatch_query_y.shape[0], minibatch_query_y.shape[1] + 1, minibatch_query_y.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root
    minibatch_decode_seq_query[:, 1: ,:] = minibatch_query_y

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id
    minibatch_decode_seq_query[:, 0, :] = config.start_id

    return minibatch_x_root, minibatch_x_neighbour, minibatch_features, minibatch_decode_seq, minibatch_target_seq, minibatch_query_x, minibatch_decode_seq_query


def get_minibatch_4_test_query(root_data, query_data, path, pathlist, cstep):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_query_x = list()
    minibatch_query_y = list()

    for o in range(config.batch_size):
        baseloc = o + cstep * config.batch_size
        minibatch_x_root.append(root_data[path, baseloc : baseloc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])
        minibatch_query_x.append(query_data[pathlist[path]][baseloc : baseloc + config.in_seq_length, :])
        minibatch_query_y.append(query_data[pathlist[path]][baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_query_x = np.stack(minibatch_query_x)
    minibatch_query_y = np.stack(minibatch_query_y)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_decode_seq_query = np.zeros((minibatch_query_y.shape[0], minibatch_query_y.shape[1] + 1, minibatch_query_y.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root
    minibatch_decode_seq_query[:, 1: ,:] = minibatch_query_y

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id
    minibatch_decode_seq_query[:, 0, :] = config.start_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq, minibatch_query_x, minibatch_decode_seq_query

def get_minibatch_4_test_all_comb(root_data, neighbour_data, features_info, features_time, query_data, path, pathlist, cstep):
    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_query_x = list()
    minibatch_query_y = list()
    minibatch_x_neighbour = list()
    minibatch_features = list()

    for o in range(config.batch_size):
        baseloc = o + cstep * config.batch_size
        minibatch_x_root.append(root_data[path, baseloc : baseloc + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])
        minibatch_query_x.append(query_data[pathlist[path]][baseloc : baseloc + config.in_seq_length, :])
        minibatch_query_y.append(query_data[pathlist[path]][baseloc + config.in_seq_length : baseloc + config.in_seq_length + config.out_seq_length, :])
        minibatch_x_neighbour.append(neighbour_data[path, baseloc : baseloc + config.in_seq_length, :])

        f = np.zeros([config.out_seq_length + 1, config.dim_features])
        for fi in range(config.out_seq_length):
            f[fi, :config.dim_features_info] = features_info[path, :]
            f[fi, -config.dim_features_time:] = features_time[-config.valid_length + baseloc + config.in_seq_length + fi, :]
        minibatch_features.append(f)


    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    minibatch_query_x = np.stack(minibatch_query_x)
    minibatch_query_y = np.stack(minibatch_query_y)
    minibatch_x_neighbour = np.stack(minibatch_x_neighbour)
    minibatch_features = np.stack(minibatch_features)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_decode_seq_query = np.zeros((minibatch_query_y.shape[0], minibatch_query_y.shape[1] + 1, minibatch_query_y.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root
    minibatch_decode_seq_query[:, 1: ,:] = minibatch_query_y

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id
    minibatch_decode_seq_query[:, 0, :] = config.start_id

    return minibatch_x_root, minibatch_x_neighbour, minibatch_features, minibatch_decode_seq, minibatch_target_seq, minibatch_query_x, minibatch_decode_seq_query

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

def get_event_filter(event_period):
    event_filter = [0] * config.valid_length
    for etime in event_period:
        for t in range(etime[0], etime[1]):
            vt = t - (config.full_length - config.valid_length)
            if vt < 0:
                break
            event_filter[vt] = 1
    return event_filter

def get_event_filter_allpath(event_data, pathlist):
    filter = np.zeros((len(pathlist), config.valid_length))
    for idx, path in enumerate(pathlist):
        filter[idx] = get_event_filter(event_data[path])
        # print(np.sum(filter[idx]))
    # print(filter.shape)
    return filter

def get_event_orders(event_filter_allpath, full_train_order, num_seq, tsteps=500):
    train_order = [0] * (config.batch_size * tsteps)
    curnum = 0
    for ord in full_train_order:
        seq_id = ord // num_seq
        seq_loc = ord % num_seq
        if np.sum(event_filter_allpath[seq_id, seq_loc + config.in_seq_length : seq_loc + config.in_seq_length + config.out_seq_length]) > 0:
            train_order[curnum] = ord
            curnum += 1
        if curnum == config.batch_size * tsteps:
            break
    assert curnum == config.batch_size * tsteps
    return train_order

def get_minibatch_4_test_event(root_data, event_filter, path, startidx, neighbour_data=None):
    assert root_data.shape[1] == len(event_filter)

    minibatch_x_root = list()
    minibatch_y_root = list()
    minibatch_x_neighbour = list()

    curidx = startidx
    while len(minibatch_x_root) < config.batch_size and \
                    curidx < config.valid_length - config.in_seq_length - config.out_seq_length:
        flag = True
        for i in range(config.out_seq_length):
            if event_filter[curidx + config.in_seq_length + i] == 0:
                flag = False
                break
        if not flag:
            curidx += 1
            continue
        minibatch_x_root.append(root_data[path, curidx : curidx + config.in_seq_length, :])
        minibatch_y_root.append(root_data[path, curidx + config.in_seq_length : curidx + config.in_seq_length + config.out_seq_length, :])
        # minibatch_x_neighbour.append(neighbour_data[path, baseloc : baseloc + config.in_seq_length, :])
        curidx += 1


    if len(minibatch_x_root) < config.batch_size / 2:
        return None, None, None, curidx, True
    else:
        for last in range(config.batch_size - len(minibatch_x_root)):
            minibatch_x_root.append(minibatch_x_root[-1])
            minibatch_y_root.append(minibatch_y_root[-1])

    minibatch_x_root = np.stack(minibatch_x_root)
    minibatch_y_root = np.stack(minibatch_y_root)
    # minibatch_x_neighbour = np.stack(minibatch_x_neighbour)

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1: ,:] = minibatch_y_root
    minibatch_target_seq[:, :-1 ,:] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    if curidx < config.valid_length - config.in_seq_length - config.out_seq_length:
        return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq, curidx, False

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq, curidx, True

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

def load_event_data():
    '''

    rootdata, neighbourdata, pathlist = load_data(5, 5)
    event_period = pickle.load(open(config.data_path + "event_link_set_beijing_event_time_1km.pkl", "rb"))

    pathid = dict()
    # dup = 0
    for idx ,path in enumerate(pathlist):
        # if path in pathid:
        #     dup += 1
        pathid[path] = idx
    # print(dup)

    event_period_data_list = list()
    for path in event_period:
        if path not in pathid:
            continue
        for event in event_period[path]:
            starttime = event[0]
            endtime = event[1]
            if endtime < config.full_length - config.valid_length:
                continue
            for ctime in range(starttime, endtime - config.out_seq_length):
                event_period_data_list.append(rootdata[pathid[path], ctime - config.in_seq_length : ctime + config.out_seq_length, :])

    event_period_data_list = np.stack(event_period_data_list, axis=0)
    print(event_period_data_list.shape)
    return event_period_data_list
    '''
    event_period = pickle.load(open(config.data_path + "event_link_set_beijing_event_time_1km.pkl", "rb"))
    return event_period

def get_query_data():
    print("Loading Query %d..." % config.impact_k)
    data = pickle.load(open(config.data_path + "query_distribution_beijing_1km_k_%d_filtfilt.pkl" % config.impact_k, "rb"), encoding='latin1')
    for node in data:
        data[node] = np.expand_dims(np.array(data[node]), axis=1)
        assert data[node].shape[0] == config.full_length
        assert data[node].shape[1] == 1
    print("Query Loaded")
    return data


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
    '''
    fi, ft, fp = load_features(["1525826704", "1561981475"])
    print(fi)
    print(ft)
    print(fp)
    '''
    # load_features()
    '''
    e = load_event_data()
    for node in e.keys():
        road = node
        break
    print(e[road])
    ef = get_event_filter(e[road])
    print(ef)
    print(len(ef))
    exit()
    x, d, t, eidx, end = get_minibatch_4_test_event(r[:, -config.valid_length:,:], ef, 0, 0)
    print(x.shape)
    print(d.shape)
    print(t.shape)
    print(eidx)
    print(end)
    '''
    '''
    e = load_event_data()
    p = get_pathlist()
    eap = get_event_filter_allpath(e, p)

    each_num_seq = config.valid_length - (config.in_seq_length + config.out_seq_length) + 1
    total_batch_size = 15073 * each_num_seq

    eorder = get_event_orders(eap, list(range(total_batch_size)), each_num_seq)
    print(len(eorder))

    for i in range(10):
        pathid = eorder[i] // each_num_seq
        pathlod = eorder[i] % each_num_seq
        print(pathid)
        print(pathlod)

        print(eap[pathid][pathlod + config.in_seq_length: pathlod + config.in_seq_length + config.out_seq_length])
    '''
    get_query_data()



    pass

