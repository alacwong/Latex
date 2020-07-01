# Module to generate dataset
import os
from PIL import Image
import random
import math

path = 'YOUR PATH'


def main():
    """
    Use pillows to merge together images while generating bounding boxes
    """
    # path to data set
    labels = []
    for label in os.listdir(path):  # generate list of images per label
        labels.append(os.listdir(path + '\\' + label))

    n = 9  # pick a perfect square
    # TODO distribute label on  test images
    data = distribute_data(labels, n)
    annotations = paste(data)
    # TODO save annotations to JSON so we can use it later for ML


def distribute_data(labels, n):
    """
    distribute data to images
    HAVE VARIANCE
    DO NOT BE LAZY AND CREATE A TEST IMAGE WITH ALL THE SAME LABELS
    this part is cancer, so im not doing it hehexd
    :return: [ [[label, num], ['t', '50'], ['x', '60'] ....],
    [['5', 8], ['9', 3], ['-', '4'] ... ]  ...], each list should be length n
    """

    return [[[]]]


def paste(data):
    """
    Paste labels onto image
    :return: annotated obj filename: class_label/bounding box
    """
    counter = 0

    masks = generate_masks(len(data[0]))
    annotations = {}

    for image in data:
        canvas = Image.new('RGB', (1024, 1024), (255, 255, 255))  # white canvas
        annotations['IMAGE-' + counter] = []
        for i in len(image):
            img_label = Image.open(path + '\\' + image[i][0] + '\\' + image[i][1])
            canvas.paste(img_label.resize((masks[i][2] - masks[i][0],
                                           masks[i][2] - masks[i][0])), box=masks[i])
            annotations['IMAGE-' + counter].append({image[i][0], masks[i]})  # save bounding box

        # save to data set/generated_data (pls save to dataset directory, however u an change the name
        # for 'generated_data')
        canvas.save('../../dataset/generated_data/IMAGE-' + counter)

    return annotations


def generate_masks(n):
    """
    generate starting coords
    :return: 1D array of coord start positions
    """

    masks = []  # generate permutations
    for i in range(int(math.sqrt(n))):
        for j in range(int(math.sqrt(n))):
            masks.append((i, j))

    for x in range(len(masks)):  # generate masks
        [i, j] = masks[x]
        scale = random.uniform(1, 1024 / math.sqrt(n) / 45)
        masks[x] = (int(i * 1024 / math.sqrt(n)),
                    int(j * 1024 / math.sqrt(n)),
                    int(i * 1024 / math.sqrt(n)) + 45 * scale,
                    int(j * 1024 / math.sqrt(n)) + 45 * scale)
    return masks


if __name__ == '__main__':
    main()
