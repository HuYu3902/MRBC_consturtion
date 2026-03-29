'''
Importing third-party libraries
'''
import os
import csv
import glob
import torch
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset as Image_Dataset
from torch_geometric.data import Data as Graph_Data
from torch_geometric.data import Dataset as Graph_Dataset

'''
Importing internal libraries
'''
import utils
from config import args


class brian_image_dataset(Image_Dataset):
    def __init__(self, data_load_dir, data_load_csv, data_dir):
        super(brian_image_dataset, self).__init__()

        '''
        Initialization
        '''
        self.data_load_dir = data_load_dir
        self.data_load_csv = data_load_csv
        self.data_dir = data_dir
        self.data_list = self.load_csv()
        self.mask_generator = utils.MaskGenerator3D(input_size=args.input_size, mask_patch_size=args.mask_patch_size, mask_ratio=args.mask_ratio)

    '''
    Loading paths of all data into a list
    '''
    def load_csv(self):
        subject_paths = []
        if not os.path.exists(os.path.join(self.data_load_dir, self.data_load_csv)):
            for subject in sorted(os.listdir(self.data_dir)):
                subject_paths += glob.glob(os.path.join(self.data_dir, subject))
            random.shuffle(subject_paths)

            # Create the CSV and write the paths into it
            with open(os.path.join(self.data_load_dir, self.data_load_csv), mode='w', newline='') as f:
                writer = csv.writer(f)
                for subject_path in subject_paths:
                    subject_ID = os.path.split(subject_path)[-1].split('.')[0]
                    writer.writerow([subject_ID, subject_path])

        # Write the paths from the CSV into a list
        data_list = []
        with open(os.path.join(self.data_load_dir, self.data_load_csv)) as f:
            reader = csv.reader(f)
            for row in reader:
                subject_ID, subject_path = row
                pair_dict = {'subject_ID': subject_ID, 'subject_path': subject_path}
                data_list.append(pair_dict)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        subject_ID = self.data_list[index]['subject_ID']
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.data_list[index]['subject_path']))[np.newaxis, ...]
        mask = self.mask_generator.generate_mask()[np.newaxis, ...]
        return subject_ID, image, mask
