# Library to recognize the spacial relationship between characters
# types of relationship
# over/under (fraction)
# subscript x^i
# concat 345
# left 4 5 6
# in sqrt(x)

# given a mathematical expression, predict its spatial relationship with each symbol
from sklearn import svm


def train(data, spatial_relationship, symbol):
    """
    train svm with data and then save
    data: 8 vector [ [bounding box of parent] + [bounding box symbol of child] ]
    spatial_relationship: relationship between parent and child
    symbol: denote type of classifier
    :return:
    """
    classifier = svm.SVC()
    classifier.fit(data, spatial_relationship)


def predict(symbols, new_symbol):
    """
    given a dictionary of symbols, go through each symbol and determine its
    relationship with new_symbol
    :return: dictionary symbols with their label and confidence on label
    """