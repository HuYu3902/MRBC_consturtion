'''
Importing third-party libraries
'''
import os
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from monai.utils import set_determinism

'''
Importing internal libraries
'''
import utils
from dataloader import brian_image_dataset
from net.rec.maskConvNet import Feature_Extraction
from config import args


def main():
    '''
    Initialize model
    '''
    net = Feature_Extraction(hidden_size=args.hidden_size).to(device)
    save_model = torch.load(os.path.join(args.feature_learning_model_dir, 'model_100.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    net.eval()

    '''
    Loading atlas
    '''
    atlas_s = np.load(args.atlas_s)
    atlas_m = np.load(args.atlas_m)

    '''
    Loading data
    '''
    dataset = brian_image_dataset(data_load_dir=args.brain_network_construction_data_load_dir,
                                  data_load_csv='brain_network_construction_data.csv',
                                  data_dir=args.brain_network_construction_data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    '''
    Brain network construction
    '''
    for subject_ID, image, _ in loader:
        image = image.to(device).float()

        '''
        Get feature maps from network
        '''
        feature_map_s, feature_map_m = net(image)
        feature_map_s = feature_map_s.cpu().detach().numpy().squeeze()
        feature_map_m = feature_map_m.cpu().detach().numpy().squeeze()

        '''
        Get ROI feature as node feature
        '''
        roi_feature = utils.get_roi_feature(feature_map_s, feature_map_m, atlas_s, atlas_m)
        os.makedirs(os.path.join(args.AD_diagnosis_brain_network_dir, subject_ID[0]), exist_ok=True)
        np.savetxt(os.path.join(args.AD_diagnosis_brain_network_dir, subject_ID[0], args.atlas + '_node_feature.txt'), roi_feature)

        '''
        Get ROI feature as adjacency matrix
        '''
        adjacency_matrix = utils.get_adjacency_matrix(roi_feature)
        np.savetxt(os.path.join(args.AD_diagnosis_brain_network_dir, subject_ID[0], args.atlas + '_adjacency_matrix.txt'), adjacency_matrix)

        print('Completed brain network construction: ', subject_ID[0])


if __name__ == "__main__":
    '''
    Ignoring warnings
    '''
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    '''
    GPU configuration
    '''
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    GPU_avai = torch.cuda.is_available()
    print('If the GPU is available? ' + str(GPU_avai))
    GPU_iden = args.gpu
    device = torch.device('cuda:{}'.format(GPU_iden) if GPU_avai else 'cpu')
    print('Currently using: GPU #' + str(GPU_iden))

    '''
    Random seed
    '''
    set_determinism(seed=args.seed)

    '''
    Setting cudnn to benchmark
    '''
    cudnn.benchmark = True

    '''
    Main
    '''
    main()
