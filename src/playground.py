
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import progressbar
import networkx as nx

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
    eventsetfilename = datapath + "event_link_set_beijing_1km"
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
    resultfilename = resultspath + "event_link_set_beijing_link_1km.txt"
    resultfile = open(resultfilename, "w")
    for link in linklist:
        resultfile.write(str(link))
        resultfile.write("\n")
    resultfile.close()

def draw_roadnet():
    roadnetfilename = resultspath + "event_link_set_beijing_link_1km.txt"
    roadnetfile = open(roadnetfilename, "r")

    # prfilename = resultspath + "pagerank_1km.txt"
    # prfile = open(prfilename, "w")
    bar = progressbar.ProgressBar(max_value=1151)
    for iter, line in enumerate(roadnetfile):
        bar.update(iter)

        content = eval(line)

        graph = nx.DiGraph()
        linkdict = dict()
        for link in content:
            if link[2] == 1:
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
        nx.draw(graph, node_size=nsize)
        # nx.draw(graph, node_size=20)
        '''

        import operator
        sortedlist = list()
        for node, value in sorted(prnodes.items(), key=operator.itemgetter(1), reverse=True):
            sortedlist.append((node, value))
        # prfile.write(str(sortedlist))
        # prfile.write("\n")

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
        exit()

    roadnetfile.close()
    # prfile.close()

def get_data():
    import pickle

    d = pickle.load(open(datapath + "event_traffic_completion_beijing_15min.pkl", "rb"), encoding='latin1')
    print(d)


if __name__ == "__main__":
    # roadnet_extraction()
    # draw_roadnet()
    get_data()