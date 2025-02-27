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


class brain_network_dataset_diagnosis(Graph_Dataset):
    def __init__(self, data_load_dir, data_load_csv, data_dir, k_num):
        super(brain_network_dataset_diagnosis, self).__init__()

        '''
        Initialization
        '''
        self.data_load_dir = data_load_dir
        self.data_load_csv = data_load_csv
        self.data_dir = data_dir
        self.k_num = k_num

        self.name2label = {}
        for name in sorted(os.listdir(self.data_dir)):
            self.name2label[name] = len(self.name2label.keys())

        self.data_list = self.load_csv()

    '''
    Loading paths of all data into CSV
    '''
    def load_csv(self):
        if not os.path.exists(os.path.join(self.data_load_dir, self.data_load_csv)):
            # Loading paths of all data into a list
            subject_paths = []
            for classification_name in self.name2label.keys():
                for subject_ID in sorted(os.listdir(os.path.join(self.data_dir, classification_name))):
                    subject_paths += glob.glob(os.path.join(self.data_dir, classification_name, subject_ID))
            random.shuffle(subject_paths)

            # Create the CSV and write the paths into it
            with open(os.path.join(self.data_load_dir, self.data_load_csv), mode='w', newline='') as f:
                writer = csv.writer(f)
                for subject_path in subject_paths:
                    subject_ID = os.path.split(subject_path)[-1]
                    node_feature_path = os.path.join(subject_path, "AAL_node_feature.txt")
                    adjacency_matrix_path = os.path.join(subject_path, "AAL_adjacency_matrix.txt")
                    classification_name = subject_path.split(os.sep)[-2]
                    label = self.name2label[classification_name]
                    writer.writerow([subject_ID, node_feature_path, adjacency_matrix_path, label])

        # Write the paths from the CSV into a list
        data_list = []
        with open(os.path.join(self.data_load_dir, self.data_load_csv)) as f:
            reader = csv.reader(f)
            for row in reader:
                subject_ID, node_feature_path, adjacency_matrix_path, label = row
                label = int(label)
                pair_dict = {'subject_ID': subject_ID, 'node_feature_path': node_feature_path, 'adjacency_matrix_path': adjacency_matrix_path, 'label': label}
                data_list.append(pair_dict)
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, index):
        subject_ID = self.data_list[index]['subject_ID']
        node_feature = np.loadtxt(self.data_list[index]['node_feature_path'])
        adjacency_matrix = np.loadtxt(self.data_list[index]['adjacency_matrix_path'])
        label = self.data_list[index]['label']

        adjacency_matrix = utils.get_binary_matrix(adjacency_matrix, k_num=self.k_num)

        adjacency_matrix = np.float32(adjacency_matrix)
        node_feature = np.float32(node_feature)

        edge_index = utils.get_binary_edge(adjacency_matrix)

        node_feature = torch.from_numpy(node_feature)
        edge_index = torch.from_numpy(edge_index)

        return subject_ID, Graph_Data(x=node_feature, y=label, edge_index=edge_index)


class brain_network_dataset_prediction(Graph_Dataset):
    def __init__(self, data_load_dir, data_load_csv, data_dir, k_num, phenotypic_information_csv, predicted_label_name):
        super(brain_network_dataset_prediction, self).__init__()

        '''
        Initialization
        '''
        self.data_load_dir = data_load_dir
        self.data_load_csv = data_load_csv
        self.data_dir = data_dir
        self.k_num = k_num
        self.phenotypic_information_csv = phenotypic_information_csv
        self.predicted_label_name = predicted_label_name
        self.data_list = self.load_csv()

    '''
    Loading paths of all data into CSV
    '''
    def load_csv(self):
        if not os.path.exists(os.path.join(self.data_load_dir, self.data_load_csv)):
            df = pd.read_csv(self.phenotypic_information_csv)

            # Loading paths of all data into a list
            subject_paths = []
            for subject_ID in sorted(os.listdir(self.data_dir)):
                subject_paths += glob.glob(os.path.join(self.data_dir, subject_ID))
            random.shuffle(subject_paths)

            # Create the CSV and write the paths into it
            with open(os.path.join(self.data_load_dir, self.data_load_csv), mode='w', newline='') as f:
                writer = csv.writer(f)
                for subject_path in subject_paths:
                    subject_ID = os.path.split(subject_path)[-1].split('ADNI_')[-1]
                    node_feature_path = os.path.join(subject_path, "AAL_node_feature.txt")
                    adjacency_matrix_path = os.path.join(subject_path, "AAL_adjacency_matrix.txt")
                    predicted_label = df[df['PTID'].astype(str) == subject_ID][self.predicted_label_name]
                    if predicted_label.empty or pd.isnull(predicted_label.iloc[0]) or not self.is_valid_number(predicted_label.iloc[0]):
                        continue
                    writer.writerow([subject_ID, node_feature_path, adjacency_matrix_path, predicted_label.iloc[0]])

        # Write the paths from the CSV into a list
        data_list = []
        with open(os.path.join(self.data_load_dir, self.data_load_csv)) as f:
            reader = csv.reader(f)
            for row in reader:
                subject_ID, node_feature_path, adjacency_matrix_path, predicted_label = row
                pair_dict = {'subject_ID': subject_ID, 'node_feature_path': node_feature_path, 'adjacency_matrix_path': adjacency_matrix_path, 'predicted_label': float(predicted_label)}
                data_list.append(pair_dict)
        return data_list

    def is_valid_number(self, value):
        """
      Check if the value is a valid number (i.e., not NaN, None, or a non-numeric string).
      """
        try:
            # Try to convert the value to a float
            float(value)
            return True
        except (ValueError, TypeError):
            # If conversion fails, it's not a valid number
            return False

    def len(self):
        return len(self.data_list)

    def get(self, index):
        subject_ID = self.data_list[index]['subject_ID']
        node_feature = np.loadtxt(self.data_list[index]['node_feature_path'])
        adjacency_matrix = np.loadtxt(self.data_list[index]['adjacency_matrix_path'])
        predicted_label = self.data_list[index]['predicted_label']

        adjacency_matrix = np.abs(adjacency_matrix)

        adjacency_matrix = np.float32(adjacency_matrix)
        node_feature = np.float32(node_feature)

        edge_index, edge_attr = utils.get_weight_edge(adjacency_matrix)

        node_feature = torch.from_numpy(node_feature)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)

        return subject_ID, Graph_Data(x=node_feature, y=predicted_label, edge_attr=edge_attr, edge_index=edge_index)
