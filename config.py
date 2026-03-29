import argparse


parser = argparse.ArgumentParser('MRBC script', add_help=False)


'''
common parameters
'''
parser.add_argument("--num_workers", type=int, help="number of workers",
                    dest="num_workers", default=4)
parser.add_argument("--gpu", type=int, help="gpu id",
                    dest="gpu", default='0')
parser.add_argument("--seed", type=int, help="seed",
                    dest="seed", default='1234')
parser.add_argument("--hidden_size", type=int, help="hidden size",
                    dest="hidden_size", default=128)


'''
feature learning parameters
'''
parser.add_argument("--feature_learning_data_load_dir", type=str, help="data load folder with feature learning",
                    dest="feature_learning_data_load_dir", default=r"./data_load/feature_learning")
parser.add_argument("--feature_learning_data_dir", type=str, help="feature learning data folder",
                    dest="feature_learning_data_dir", default=r"../../datasets_and_preprocessing/preprocessed/feature_learning_data")
parser.add_argument("--feature_learning_log_dir", type=str, help="log folder with feature learning",
                    dest="feature_learning_log_dir", default=r'./log/feature_learning')
parser.add_argument("--input_size", type=int, help="input_size",
                    dest="input_size", default=(160, 192, 160))
parser.add_argument("--mask_patch_size", type=int, help="mask_patch_size",
                    dest="mask_patch_size", default=(32, 32, 32))
parser.add_argument("--mask_ratio", type=float, help="mask_ratio",
                    dest="mask_ratio", default=0.6)
parser.add_argument("--feature_learning_batch_size", type=int, help="batch size with feature learning",
                    dest="feature_learning_batch_size", default=4)
parser.add_argument("--feature_learning_lr", type=float, help="learning rate with feature learning",
                    dest="feature_learning_lr", default=1e-4)
parser.add_argument("--feature_learning_weight_decay_factor", type=float, help="weight decay factor with feature learning",
                    dest="feature_learning_weight_decay_factor", default=1e-4)
parser.add_argument("--feature_learning_epochs", type=int, help="number of epochs with feature learning",
                    dest="feature_learning_epochs", default=100)
parser.add_argument("--feature_learning_n_val_iter", type=int, help="number valuation iter with feature learning",
                    dest="feature_learning_n_val_iter", default=1)
parser.add_argument("--feature_learning_model_dir", type=str, help="model folder with feature learning",
                    dest="feature_learning_model_dir", default=r'./model/feature_learning')


'''
brain network construction parameters
'''
parser.add_argument("--atlas", type=str, help="atlas",
                    dest="atlas", default=r"AAL")
parser.add_argument("--atlas_s", type=str, help="atlas_s",
                    dest="atlas_s", default=r"./atlas/AAL/20x24x20.npy")
parser.add_argument("--atlas_m", type=str, help="atlas_m",
                    dest="atlas_m", default=r"./atlas/AAL/40x48x40.npy")
parser.add_argument("--brain_network_construction_data_load_dir", type=str, help="data load folder with brain network construction",
                    dest="brain_network_construction_data_load_dir", default=r"./data_load/brain_network_construction/CoRR")
parser.add_argument("--brain_network_construction_data_dir", type=str, help="feature learning data folder",
                    dest="brain_network_construction_data_dir", default=r"../../datasets_and_preprocessing/preprocessed/CoRR")


args = parser.parse_args()
