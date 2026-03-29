'''
Importing third-party libraries
'''
import heapq
import numpy as np
import SimpleITK as sitk
from ptflops import get_model_complexity_info
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim


'''
Getting roi feature
'''
def get_roi_feature(feature_map_s, feature_map_m, template_s, template_m):
    roi_feature = list()
    # special ROIs are small brain regions that may be ignored on the downsampled template
    special_ROI = [17, 18]

    # 90 ROIs
    for i in range(1, 95):
        # feature aggregation at different scales
        if i not in special_ROI:
            roi_template = (template_s == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_s, axis=(1, 2, 3)) / np.sum(roi_template)
            roi_feature.append(feature)
        else:
            roi_template = (template_m == i).astype(np.uint8)[np.newaxis, :, :, :]
            feature = np.sum(roi_template * feature_map_m, axis=(1, 2, 3)) / np.sum(roi_template)
            roi_feature.append(np.concatenate((feature, feature), axis=0))

    roi_feature = np.array(roi_feature)
    return roi_feature


'''
Getting adjacency matrix
'''
def get_adjacency_matrix(roi_feature):
    adjacency_matrix = cosine_similarity(roi_feature)

    return adjacency_matrix


'''
Getting preprocessed adjacency matrix
'''
def get_binary_matrix(connection_matrix, k_num):
    roi_num = connection_matrix.shape[0]

    # get sparse and binary connections
    for i in range(roi_num):
        node_connection = connection_matrix[i, :]
        # choose the k closest positions but exclude the node itself
        position = heapq.nlargest(k_num + 1, range(len(node_connection)), node_connection.__getitem__)
        sparse_connection = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k_num + 1):
            sparse_connection[position[j]] = 1
        sparse_connection[i] = 0
        connection_matrix[i, :] = sparse_connection

    # complete connection matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if connection_matrix[i, j] == 1:
                connection_matrix[j, i] = 1

    return connection_matrix


def get_sparse_matrix(connection_matrix, k_num):
    roi_num = connection_matrix.shape[0]

    # get sparse connections
    for i in range(roi_num):
        node_connection = connection_matrix[i, :]
        # choose the k closest positions but exclude the node itself
        position = heapq.nlargest(k_num + 1, range(len(node_connection)), node_connection.__getitem__)
        sparse_connection = np.zeros(roi_num)
        for j in range(k_num + 1):
            sparse_connection[position[j]] = connection_matrix[i, position[j]]
        sparse_connection[i] = 0
        connection_matrix[i, :] = sparse_connection

    # complete connection matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if connection_matrix[i, j] != 0:
                connection_matrix[j, i] = connection_matrix[i, j]
    return connection_matrix


'''
Getting edge
'''
def get_binary_edge(adjacency_matrix):
    edge = list()
    roi_num = adjacency_matrix.shape[0]

    # save edge by [Source Node, Target Node] from adjacency matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if adjacency_matrix[i, j] == 1:
                edge.append(np.array([i, j]))

    # transpose edge list for graph construction
    edge = np.swapaxes(np.array(edge), axis1=0, axis2=1)

    return edge


def get_weight_edge(adjacency_matrix):
    edge_idx_lst = list()
    edge_attr_lst = list()
    roi_num = adjacency_matrix.shape[0]

    # save edge by [Source Node, Target Node] from adjacency matrix
    for i in range(roi_num):
        for j in range(i + 1, roi_num):
            if adjacency_matrix[i, j] != 0:
                edge_idx_lst.append(np.array([i, j]))
                edge_attr_lst.append(adjacency_matrix[i, j])

    # transpose edge list for graph construction
    edge_inx = np.swapaxes(np.array(edge_idx_lst), axis1=0, axis2=1)
    edge_attr = np.array(edge_attr_lst)

    return edge_inx, edge_attr


class MaskGenerator3D:
    def __init__(self, input_size=(160, 192, 160), mask_patch_size=(32, 32, 32), mask_ratio=0.6):
        self.input_size = input_size  # Input size of the 3D image (depth, height, width)
        self.mask_patch_size = mask_patch_size  # Size of the mask patches (depth, height, width)
        self.mask_ratio = mask_ratio  # Ratio of the image to be masked

        # Ensure that the input size is divisible by the mask patch size
        assert self.input_size[0] % self.mask_patch_size[0] == 0
        assert self.input_size[1] % self.mask_patch_size[1] == 0
        assert self.input_size[2] % self.mask_patch_size[2] == 0

        # Calculate the number of patches in each dimension
        self.rand_size = (self.input_size[0] // self.mask_patch_size[0],
                          self.input_size[1] // self.mask_patch_size[1],
                          self.input_size[2] // self.mask_patch_size[2])

        # Total number of patches and the number of patches to be masked
        self.token_count = self.rand_size[0] * self.rand_size[1] * self.rand_size[2]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))  # Calculate the number of patches to mask

    def generate_mask(self):
        # Randomly generate indices for patches to be masked
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.ones(self.token_count, dtype=int)  # Default all to 1
        mask[mask_idx] = 0  # Set the patches to be masked as 0

        # Reshape the mask into a 3D shape matching the patch grid
        mask = mask.reshape(self.rand_size)

        # Expand the mask to match the original input size using repeat
        mask_expanded = mask.repeat(self.mask_patch_size[0], axis=0).repeat(self.mask_patch_size[1], axis=1).repeat(self.mask_patch_size[2], axis=2)

        return mask_expanded
