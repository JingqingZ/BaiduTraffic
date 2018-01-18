
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import progressbar

datapath = "../../data/"

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
    print("Number of Previous:", nprev)
    print("Number of Next", nnext)
    print("Avg Prev %.2f, Next %.2f" % (sump / len(node_dict.keys()), sumn / len(node_dict.keys())))

    print("Getting links ...")
    eventsetfilename = datapath + "event_link_set_beijing.txt"
    eventsetfile = open(eventsetfilename, "r")
    bar = progressbar.ProgressBar(max_value=1151)
    linklist = list()
    for iter, line in enumerate(eventsetfile):
        bar.update(iter)
        nodeids = line.replace("\n", "").split("\t")

        clink = list()
        for nid in nodeids:
            for njd in nodeids:
                if nid == njd:
                    continue
                if nid in node_dict and njd in node_dict[nid].next:
                    clink.append((nid, njd))
                if nid in node_dict and njd in node_dict[nid].previous:
                    clink.append((njd, nid))
        linklist.append(clink)
    eventsetfile.close()

    print("Saving ... ")
    resultfilename = "event_link_set_beijing_link.txt"
    resultfile = open(resultfilename, "w")
    for link in linklist:
        resultfile.write(str(link))
        resultfile.write("\n")
    resultfile.close()


if __name__ == "__main__":
    roadnet_extraction()