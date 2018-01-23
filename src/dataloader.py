
import pickle
import queue
import numpy as np
import progressbar

import config

def find_neighbours(predecessor=5, successors=5):

    event_filter_file = open(config.data_path + "event_filter.txt", "r")
    event_filter_flag = eval(event_filter_file.readline())
    event_filter_file.close()

    event_set_file = open(config.data_path + "event_link_set_beijing", "r")
    event_set = event_set_file.readlines()
    event_set_file.close()

    event_link_file = open(config.result_path + "event_link_set_beijing_link.txt", "r")
    event_link = event_link_file.readlines()
    event_link_file.close()

    event_pagerank_file = open(config.result_path + "pagerank.txt", "r")
    event_pagerank = event_pagerank_file.readlines()
    event_pagerank_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    outfile = open(config.result_path + "neighbours.txt", "w")

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
    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    neighbour_file = open(config.result_path + "neighbours.txt", "r")
    neighbour = neighbour_file.readlines()

    rootdata = list()
    neigdata = list()
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

    rootdata = np.stack(rootdata)
    neigdata = np.stack(neigdata)

    assert rootdata.shape[:-1] == neigdata.shape[:-1]

    print("Data Loaded: x_root %s, x_neighbour: %s" % (rootdata.shape, neigdata.shape))

    return rootdata, neigdata


def get_minibatch(root_data, neighbour_data, order, num_seq):
    minibatch_x_root = list()
    minibatch_x_neighbour = list()
    minibatch_y_root = list()

    for o in order:
        seq_id = o // num_seq
        seq_loc = o % num_seq
        minibatch_x_root.append(root_data[seq_id, seq_loc : seq_loc + config.seq_length, :])
        minibatch_y_root.append(root_data[seq_id, seq_loc + config.seq_length : seq_loc + 2 * config.seq_length, :])
        minibatch_x_neighbour.append(neighbour_data[seq_id, seq_loc : seq_loc + config.seq_length, :])

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


if __name__ == "__main__":
    # find_neighbours(3, 3)
    r, n = load_data(3, 3)
    get_minibatch(r, n, order=[0,1], num_seq=r.shape[1] - 2 * config.seq_length + 1)
