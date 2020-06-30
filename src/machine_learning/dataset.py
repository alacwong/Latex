# Module for getting data to train

# class ImageDataset
from mrcnn.utils import Dataset
import os
import csv

class Equations(Dataset):
    """
    Dataset obj to train image data
    """
    def load_equation(self):
        """
        Load the datatset
        :return:
        """
        path = '../../dataset/images'
        self.add_class('equation', 0, 'equation')
        count = 0
        for file in os.listdir(path):
            self.add_image('dataset', )
            self.add


    def load_mask(self, image_id):
        """
        load mask from csv
        :param image_id:
        :return:
        """
