
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import progressbar
import networkx as nx
import pickle

import config
import dataloader

datapath = "../../data/"
resultspath = "../results/"

class RoadNode():
    def __init__(self, id):
        # may duplicate
        self.previous = list()
        # may duplicate
        self.next = list()
        self.id = id

def roadnet_extraction():

    roadnetfilename = datapath + "beijing roadnet/R.mid"
    roadnetfile = open(roadnetfilename, "rb")

    node_dict = dict()

    print("Loading road network ... ")
    bar = progressbar.ProgressBar(max_value=857767)
    for iter, l in enumerate(roadnetfile):
        bar.update(iter)

        line = str(l)[2:-1]
        content = line.replace("\"", "").split("\\t")

        current_node = content[1]
        previous_node = content[9]
        next_node = content[10]

        if current_node not in node_dict:
            n = RoadNode(current_node)
            node_dict[current_node] = n
        node_dict[current_node].previous.append(previous_node)
        node_dict[current_node].next.append(next_node)

        if previous_node not in node_dict:
            n = RoadNode(previous_node)
            node_dict[previous_node] = n
        node_dict[previous_node].next.append(current_node)

        if next_node not in node_dict:
            n = RoadNode(next_node)
            node_dict[next_node] = n
        node_dict[next_node].previous.append(current_node)

    roadnetfile.close()
    print("Network loaded!")

    # analysis
    nprev = np.zeros(11)
    nnext = np.zeros(11)
    sump = sumn = 0
    for node in node_dict:
        # if len(node_dict[node].previous) > 1:
        #    print(node, str(node_dict[node].previous))
        # if len(node_dict[node].next) > 1:
        #     print(node, str(node_dict[node].next))
        sump += len(node_dict[node].previous)
        sumn += len(node_dict[node].next)
        nprev[min(10, len(node_dict[node].previous))] += 1
        nnext[min(10, len(node_dict[node].next))] += 1
    np.set_printoptions(suppress=True)
    print("Number of Road: ", len(node_dict.keys()))
    print("Number of Previous:", nprev)
    print("Number of Next", nnext)
    print("Avg Prev %.2f, Next %.2f" % (sump / len(node_dict.keys()), sumn / len(node_dict.keys())))

    print("Getting links ...")
    # eventsetfilename = datapath + "event_link_set_beijing"
    eventsetfilename = datapath + "event_link_set_beijing"
    eventsetfile = open(eventsetfilename, "r")
    event_road = dict()
    linklist = list()

    '''
    bar = progressbar.ProgressBar(max_value=1151)
    for iter, line in enumerate(eventsetfile):
        bar.update(iter)
        nodeids = line.replace("\n", "").split("\t")
        for nid in nodeids:
            event_road[nid] = 1

    print("Number of event road: ", len(event_road.keys()))
    bar = progressbar.ProgressBar(max_value=len(event_road.keys()))
    for iter, nid in enumerate(event_road.keys()):
        bar.update(iter)
        for njd in event_road.keys():
            if nid == njd:
                continue
            if nid in node_dict and njd in node_dict[nid].next:
                linklist.append((nid, njd))
            if nid in node_dict and njd in node_dict[nid].previous:
                linklist.append((njd, nid))
    '''

    def deep_next(gid, depth):
        if gid not in node_dict:
            return list()
        allnext = list()
        for d in range(depth):
            if d == 0:
                allnext.append(node_dict[gid].next)
            else:
                allnext.append(list())
                for pid in allnext[d - 1]:
                    if pid in node_dict:
                        allnext[d] += node_dict[pid].next
        return allnext

    def deep_previous(gid, depth):
        if gid not in node_dict:
            return list()
        allprev = list()
        for d in range(depth):
            if d == 0:
                allprev.append(node_dict[gid].previous)
            else:
                allprev.append(list())
                for pid in allprev[d - 1]:
                    if pid in node_dict:
                        allprev[d] += node_dict[pid].previous
        return allprev

    bar = progressbar.ProgressBar(max_value=1151)
    for iter, line in enumerate(eventsetfile):
        bar.update(iter)
        nodeids = line.replace("\n", "").split("\t")

        clink = list()
        deepprev = dict()
        deepnext = dict()
        # all_neighbour = list()
        for nid in nodeids:
            # all_neighbour += node_dict[nid].next + node_dict[nid].previous
            deepprev[nid] = deep_previous(nid, 2)
            deepnext[nid] = deep_next(nid, 2)

        for nid in nodeids:
            for njd in nodeids:
                if nid == njd:
                    continue

                foundflag = False
                for idx, dn in enumerate(deepnext[nid]):
                    if njd in dn:
                        clink.append((nid, njd, idx))
                        foundflag = True
                        break

                if foundflag:
                    continue

                for idx, dn in enumerate(deepprev[njd]):
                    if nid in dn:
                        clink.append((nid, njd, idx))
                        foundflag = True
                        break

                # if nid in node_dict and njd in node_dict[nid].next:
                #     clink.append((nid, njd))
                # if nid in node_dict and njd in node_dict[nid].previous:
                #     clink.append((njd, nid))
        '''
        print(all_neighbour)
        print(len(all_neighbour))
        allin = list()
        notin = list()
        for ne in all_neighbour:
            if ne in nodeids:
                allin.append(ne)
            else:
                notin.append(ne)
        print(allin)
        print(len(notin))
        exit()
        '''
        linklist.append(clink)

    eventsetfile.close()

    print("Saving ... ")
    resultfilename = resultspath + "event_link_set_beijing_link.txt"
    resultfile = open(resultfilename, "w")
    for link in linklist:
        resultfile.write(str(link))
        resultfile.write("\n")
    resultfile.close()

def draw_roadnet():
    roadnetfilename = resultspath + "event_link_set_beijing_link.txt"
    roadnetfile = open(roadnetfilename, "r")

    import pickle
    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    # prfilename = resultspath + "pagerank.txt"
    # prfile = open(prfilename, "w")
    bar = progressbar.ProgressBar(max_value=1151)
    for iter, line in enumerate(roadnetfile):
        bar.update(iter)

        content = eval(line)

        graph = nx.DiGraph()
        linkdict = dict()
        for link in content:
            if link[2] == 1:
                if link[0] not in traffic_data or link[1] not in traffic_data:
                    continue
                graph.add_nodes_from(link[:-1])
                graph.add_edge(link[0], link[1])
                if link[0] not in linkdict:
                    linkdict[link[0]] = list()
                linkdict[link[0]].append(link[1])
        prnodes = nx.pagerank(graph, max_iter=10000, tol=1e-8)
        # prfile.write(str(prnodes))
        # prfile.write("\n")

        '''
        
        nsize = list()
        for node in graph.nodes:
            nsize.append(prnodes[node])
        nsize = np.array(nsize)
        nsize *= 8 / np.max(nsize)
        nsize = np.exp(nsize)
        nsize *= 600 / np.max(nsize)
        # nx.draw(graph, options)
        '''
        nx.draw(graph, node_size=20)
        plt.show()
        exit()
        # nx.draw(graph, node_size=20)

        import operator
        sortedlist = list()
        for node, value in sorted(prnodes.items(), key=operator.itemgetter(1), reverse=True):
            sortedlist.append((node, value))
        # prfile.write(str(sortedlist))
        # prfile.write("\n")

        '''
        selected_graph = nx.DiGraph()
        selected_nsize = list()
        for i in range(200):
            selected_graph.add_node(sortedlist[i][0])
            selected_nsize.append(sortedlist[i][1])
        for ni in selected_graph.nodes:
            for nj in selected_graph.nodes:
                if ni == nj:
                    continue
                if ni in linkdict and nj in linkdict[ni]:
                    selected_graph.add_edge(ni, nj)


        selected_nsize = np.array(selected_nsize)
        selected_nsize *= 8 / np.max(selected_nsize)
        selected_nsize = np.exp(selected_nsize)
        selected_nsize *= 600 / np.max(selected_nsize)

        nx.draw(selected_graph, node_size=selected_nsize)
        plt.show()
        # plt.savefig(resultspath + "figs/roadnet_set_1km/roadnet_%d.png" % iter)
        # plt.clf()
        '''

    roadnetfile.close()
    # prfile.close()

def get_data():
    import pickle

    t = pickle.load(open(datapath + "event_traffic_completion_beijing_15min.pkl", "rb"), encoding='latin1')
    d = pickle.load(open(datapath + "event_traffic_flag_beijing.pkl", "rb"), encoding='latin1')
    num = 0
    incom = 0
    for key in d.keys():
        num += 1 if d[key] else 0
        if d[key] and key not in t:
            print(key)
        elif d[key] and key in t:
            incom += 1
    print(num, len(d))
    print(incom)
    print(len(t))

def draw_sequence():
    root_data, neighbour_data = dataloader.load_data(3, 3)
    plt.plot(root_data[0, :96*7, 0])
    plt.show()

def filt_error():
    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt_0.05.pkl", "rb")
    traffic_data_filt = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    sum_error = np.zeros(3)
    for key in traffic_data:
        mae = np.mean(np.abs(traffic_data[key] - traffic_data_filt[key]))
        mape = np.mean(np.abs(traffic_data[key] - traffic_data_filt[key]) / traffic_data[key])
        mse = np.mean((traffic_data[key] - traffic_data_filt[key])**2)
        sum_error[0] += mae
        sum_error[1] += mape
        sum_error[2] += mse
    sum_error /= len(traffic_data.keys())
    print(sum_error)

def compare_filt():
    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt.pkl", "rb")
    traffic_data_filt = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt_0.1.pkl", "rb")
    traffic_data_filt_01 = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min_filtfilt_0.05.pkl", "rb")
    traffic_data_filt_005 = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_completion_beijing_15min.pkl", "rb")
    traffic_data = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    traffic_data_file = open(config.data_path + "event_traffic_beijing_mv_avg_15min_completion.pkl", "rb")
    traffic_data_mv = pickle.load(traffic_data_file, encoding='latin1')
    traffic_data_file.close()

    for key in traffic_data:
        plt.plot(traffic_data[key][:96*14], color="green")
        plt.plot(traffic_data_filt[key][:96*14], color="red")
        plt.plot(traffic_data_filt_01[key][:96*14], color="orange")
        plt.plot(traffic_data_filt_005[key][:96*14], color="yellow")
        plt.plot(traffic_data_mv[key][:96*14], color="blue")
        plt.show()
        exit()

def get_event_link():
    # traffic_data_file = open(config.data_path + "event_traffic_beijing_1km_mv_avg_15min_completion.pkl", "rb")
    # traffic_data_mv = pickle.load(traffic_data_file, encoding='latin1')
    # traffic_data_file.close()

    traffic_link_set = open(config.data_path + "event_link_set_beijing_1km", "r")
    event_filter_file = open(config.data_path + "event_filter.txt", "r")
    event_filter = eval(event_filter_file.readlines()[0])

    event_time_file = open(config.data_path + "event_beijing_final.txt", "r")
    event_time = event_time_file.readlines()

    nodedict = dict()
    iter = 0
    for idx, event in enumerate(traffic_link_set):
        if event_filter[idx] == 0:
            continue
        content = event_time[iter].split("\t")
        nodes = event.replace("\n", "").split("\t")
        for node in nodes:
            if node not in nodedict:
                nodedict[node] = list()
            nodedict[node].append((int(content[0]) // 3, int(content[1]) // 3))
        iter += 1
    for idx, node in enumerate(nodedict):
        if idx > 100:
            break
        print(nodedict[node])
    pickle.dump(nodedict, open(config.data_path + "event_link_set_beijing_event_time_1km.pkl", "wb"))

def analyse_event_link():
    data = pickle.load(open(config.data_path + "event_link_set_beijing_event_time_1km.pkl", "rb"))
    for idx, node in enumerate(data):
        for time in data[node]:
            if time[1] >= 61*96:
                print(node)



if __name__ == "__main__":
    # roadnet_extraction()
    # draw_roadnet()
    # get_data()
    # draw_sequence()
    # filt_error()
    # compare_filt()
    # get_event_link()
    # analyse_event_link()
    data = pickle.load(open(datapath + "query_distribution_beijing_1km_k_50.pkl", "rb"), encoding='latin1')
    # data = np.load(config.result_path + "seq2seq_model/91_test.npz")
    # data = data["pred"]
    print(len(data.keys()))
    for node in data:
        print(data[node].shape)
        exit()
