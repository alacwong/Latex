# Library for various utilities needed in cluster
import math


def area(symbol):
    """
    Compute bounding area of symbol
    :param symbol: google vision symbol obj
    :return: float representation of area of bounding box of symbol
    """
    return (symbol.bounding_box.vertices[2].x - symbol.bounding_box.vertices[0].x) * (
            symbol.bounding_box.vertices[2].y - symbol.bounding_box[0].y)


def get_point(bounding_box):
    """
    Convert bound box to euclidean point
    :param bounding_box: bounding box for symbol
    :return: euclidean float vector
    """
    return [bounding_box.vertices[2]["x"] - bounding_box.vertices[0]["x"],
            bounding_box.vertices[3]["y"] - bounding_box.vertices[1]["y"]]


def distance(point1, point2):
    """
    Compute euclidian distance between 2 points
    :param point1: 2 tuple vector
    :param point2: 2 tuple vector
    :return: float representation of distance
    """
    return math.sqrt(math.pow((point1[0] - point2[0]), 2) +
                     math.pow(point1[1] - point2[1], 2))


def normalize(value):
    """
    turn between 0 and 1
    :param value:
    :return:
    """
    while value > 1:
        value = value / 10
    return value
