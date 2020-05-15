# Python library to find expression clusters in image
# Algorithm will be a density based clustering algorithm
# DBSCAN with parameters average  symbol distance per
# Bounding box format
# 0 - - 1
# - - - -
# 3 - - 2


import numpy as np
from sklearn.cluster import DBSCAN as DBS
import src.machine_learning.cluster_util as util


def get_cluster(eps, data, point_map):
    """
    Return list of information
    :param point_map:
    :param eps:
    :param data:
    :return:
    """
    label_map, cluster_map = {}, {}  # point to labels,  label to set of points
    classifier = DBS(eps=eps, min_samples=2).fit(data)
    for i in range(len(classifier.labels_)):
        label_map[point_map[data[i]]] = classifier.labels_[i]
    for i in range(len(data)):
        if label_map[point_map[data[i]]] not in cluster_map:
            cluster_map[label_map[point_map[data[i]]]] = set()
        cluster_map[label_map[point_map[data[i]]]].add(data[i])
    outliers = [point_map[point] for point in cluster_map[-1]]   # make each outlier its own cluster
    for i in range(len(outliers)):
        cluster_map[-i - 1] = outliers[i]
    return cluster_map


def sort_bounds(bound_map):
    """
    Sort bounded expression
    :param bound_map:
    :return:
    """
    points, point_map = [], {}
    for label in bound_map:
        p = bound_map[label][3] - bound_map[label][1] + util.normalize(bound_map[2] - bound_map[0])
        points.append(p)
        point_map[p] = label
    points.sort()
    return [point_map[p] for p in points]


def get_min_symbol(cluster_map, point_map):
    """
    Return map from cluster to smallest symbol in cluster
    :param cluster_map: map cluster label -> points
    :param point_map:  map points -> original obj
    :return: min_symbol_map: cluster label -> min symbol obj
    """
    min_symbol_map = {}
    for label in cluster_map:
        if label > 0:
            area_map = {util.area(point_map[point]): point_map[point]
                        for point in cluster_map[label]}
            area_list = [util.area(point_map[point]) for point in cluster_map[label]]
            area_list.sort()
            min_symbol_map[label] = area_map[area_list[0]]
        else:
            min_symbol_map[label] = point_map[cluster_map[label]]
    return min_symbol_map


def get_bounds(cluster_map, point_map):
    """
    Iterate through cluster, and determine a minimal bounding box and bounds entire cluster
    :param cluster_map: cluster label -> set of points
    :param point_map:   points -> vision api obj
    :return: map cluster label -> bound
    """
    bound_map = {}
    if hasattr(list(point_map.keys)[0], "bounding_box"):  # symbol
        for key in cluster_map:
            if key > 0:
                bound_map[key] = (
                    # tuple for entire cluster
                    min([symbol.bounding_box[0].x for symbol in point_map[cluster_map[key]]]),
                    min([symbol.bounding_box[0].y for symbol in point_map[cluster_map[key]]]),
                    max([symbol.bounding_box[2].x for symbol in point_map[cluster_map[key]]]),
                    max([symbol.bounding_box[2].y for symbol in point_map[cluster_map[key]]]))
            else:
                bound_map[key] = (cluster_map[key].bounding_box[0].x,
                                  cluster_map[key].bounding_box[0].y,
                                  cluster_map[key].bounding_box[2].x,
                                  cluster_map[key].bounding_box[2].y)
    else:
        for key in cluster_map:  # object
            if key > 0:
                bound_map[key] = (
                    # tuple for entire cluster
                    min([obj.boundingPoly[0].x for obj in point_map[cluster_map[key]]]),
                    min([obj.boundingPoly[0].y for obj in point_map[cluster_map[key]]]),
                    max([obj.boundingPoly[2].x for obj in point_map[cluster_map[key]]]),
                    max([obj.boundingPoly[2].y for obj in point_map[cluster_map[key]]]))
            else:
                bound_map[key] = (cluster_map[key].boundingPoly[0].x,
                                  cluster_map[key].boundingPoly[0].y,
                                  cluster_map[key].boundingPoly[2].x,
                                  cluster_map[key].boundingPoly[2].y)

    return bound_map


def process_data(symbols):
    """
    process data into  a numpy array
    keep dictionary of points to map
    :return: Numpy array of points
    """
    point_map, data = {}, []
    for symbol in symbols:
        if hasattr(symbol, "bounding_box"):
            point = util.get_point(symbol.bounding_box)
        else:
            point = util.get_point(symbol.boundingPoly)
        data.append(point)
        point_map[point] = symbol
    return [np.array(data), point_map]


def compute_epsilon(blocks):
    """
    Compute weighted average of nearest neighbor of blocks in data
    :return eps float between (0, 1):
    """
    error, total_words, block_avg, = 1.5, 0, []
    for block in blocks:
        word_list = []
        for paragraph in block.paragraph:
            for word in paragraph.words:
                word_list.append(util.get_point(word))
        dist_mat = []   # compute nearest NN matrix
        for i in range(len(word_list)):
            dist_list = []
            for j in range(len(word_list)):
                dist_list.append(util.distance(word_list[i], word_list[j]))
            dist_mat.append(dist_list)
        nn = []
        for i in range(len(dist_mat)):
            nn[i] = max(max(dist_mat[:i]), max(dist_mat[i:]))
        block_avg.append([sum(nn), len(nn)])

    total_avg = 0
    for avg in block_avg:
        total_avg += avg[0] * avg[1] / total_words
    return total_avg * error
