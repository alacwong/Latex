# Python library to find expression clusters in image
# Algorithm will be a density based clustering algorithm
# DBSCAN with parameters average  symbol distance per
import numpy as np
from sklearn.cluster import DBSCAN as DBS
import math

label_map = {}
point_map = {}


def find_clusters(blocks, symbols):
    """
    compute clusters in data
    :return: Nothing
    """

    data = process_data(symbols)
    eps = compute_epsilon(blocks)
    classifier = DBS(eps=eps, min_samples=2).fit(data)
    for i in range(len(classifier.labels_)):
        label_map[point_map[data[i]]] = classifier.labels_[i]


def process_data(symbols):
    """
    process data into  a numpy array
    keep dictionary of points to map
    :return: Numpy array of points
    """
    data = np.array()
    for symbol in symbols:
        point = get_point(symbol.bounding_box)
        data.append(point)
        point_map[point] = symbol
    return data


def get_point(bounding_box):
    """
    Convert bound box to euclidean point
    :param bounding_box: bounding box for symbol
    :return: euclidean float vector
    """
    return [bounding_box.vertices[2]["x"] - bounding_box.vertices[0]["x"],
            bounding_box.vertices[3]["y"] - bounding_box.vertices[1]["y"]]


def compute_epsilon(blocks):
    """
    Compute weighted average of nearest neighbor of blocks in data
    :return eps float between (0, 1):
    """
    error = 1.5
    total_words = 0
    block_avg = []
    for block in blocks:
        word_list = []
        for paragraph in block.paragraph:
            for word in paragraph.words:
                word_list.append(get_point(word))
        # compute nearest NN matrix
        dist_mat = []
        for i in range(len(word_list)):
            dist_list = []
            for j in range(len(word_list)):
                dist_list.append(distance(word_list[i], word_list[j]))
            dist_mat.append(dist_list)
        nn = []
        for i in range(len(dist_mat)):
            nn[i] = max(max(dist_mat[:i]), max(dist_mat[i:]))
        block_avg.append([sum(nn), len(nn)])

    total_avg = 0
    for avg in block_avg:
        total_avg += avg[0] * avg[1]/total_words
    return total_avg * error


def distance(point1, point2):
    """
    Compute euclidian distance between 2 points
    :param point1: 2 tuple vector
    :param point2: 2 tuple vector
    :return: float representation of distance
    """
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) +
                     math.pow(point1[1] - point2[1], 2))
