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


'''
save restoration image
'''
parser.add_argument("--restoration_image_data_load_dir", type=str, help="data load folder with restoration image",
                    dest="restoration_image_data_load_dir", default=r"./data_load/restoration_image")
parser.add_argument("--original_image_data_dir", type=str, help="restoration image folder",
                    dest="original_image_data_dir", default=r"./restoration_image/original_image")
parser.add_argument("--mask_image_data_dir", type=str, help="restoration image folder",
                    dest="mask_image_data_dir", default=r"./restoration_image/mask_image")
parser.add_argument("--restoration_image_data_dir", type=str, help="restoration image folder",
                    dest="restoration_image_data_dir", default=r"./restoration_image/restoration_image")


'''
small-world properties parameters
'''
parser.add_argument("--small_world_properties_brain_network_dir", type=str, help="brain network folder",
                    dest="small_world_properties_brain_network_dir", default=r'./brain_network/CoRR')


'''
AD diagnosis parameters
'''
parser.add_argument("--AD_diagnosis_brain_network_dir", type=str, help="brain network folder",
                    dest="AD_diagnosis_brain_network_dir", default=r'./brain_network/ADNI/CN')
parser.add_argument("--AD_diagnosis_data_load_dir", type=str, help="data load folder with AD diagnosis",
                    dest="AD_diagnosis_data_load_dir", default=r"./data_load/AD_diagnosis/ADNI")
parser.add_argument("--AD_diagnosis_log_dir", type=str, help="logs folder with AD diagnosis",
                    dest="AD_diagnosis_log_dir", default=r'./log/AD_diagnosis/ADNI')
parser.add_argument("--AD_diagnosis_k_num", type=int, help="k num",
                    dest="AD_diagnosis_k_num", default=8)
parser.add_argument("--AD_diagnosis_fold", type=int, help="fold",
                    dest="AD_diagnosis_fold", default=10)
parser.add_argument("--AD_diagnosis_batch_size", type=int, help="batch size",
                    dest="AD_diagnosis_batch_size", default=16)
parser.add_argument("--AD_diagnosis_lr", type=float, help="learning rate",
                    dest="AD_diagnosis_lr", default=3e-4)
parser.add_argument("--AD_diagnosis_weight_decay_factor", type=float, help="weight decay factor",
                    dest="AD_diagnosis_weight_decay_factor", default=1e-4)
parser.add_argument("--AD_diagnosis_epochs", type=int, help="number of epochs",
                    dest="AD_diagnosis_epochs", default=100)
parser.add_argument("--AD_diagnosis_n_val_epochs", type=int, help="number of epochs",
                    dest="AD_diagnosis_n_val_epochs", default=1)
parser.add_argument("--AD_weight", type=float, help="AD weight",
                    dest="AD_weight", default=0.682)
parser.add_argument("--CN_weight", type=float, help="CN weight",
                    dest="CN_weight", default=0.318)
parser.add_argument("--AD_diagnosis_model_dir", type=str, help="models folder with AD diagnosis",
                    dest="AD_diagnosis_model_dir", default=r'./model/AD_diagnosis/ADNI')


'''
prediction parameters
'''
parser.add_argument("--prediction_brain_network_dir", type=str, help="brain network folder",
                    dest="prediction_brain_network_dir", default=r'./brain_network/ADNI/all')
parser.add_argument("--prediction_data_load_dir", type=str, help="data load folder with prediction",
                    dest="prediction_data_load_dir", default=r"./data_load/prediction")
parser.add_argument("--prediction_log_dir", type=str, help="logs folder with prediction",
                    dest="prediction_log_dir", default=r'./log/prediction')
parser.add_argument("--prediction_k_num", type=int, help="k_num",
                    dest="prediction_k_num", default=8)
parser.add_argument("--phenotypic_information_csv", type=str, help="phenotypic information csv",
                    dest="phenotypic_information_csv", default=r'./phenotypic_information/phenotypic_information.csv')
parser.add_argument("--predicted_label_name", type=str, help="the predicted label name",
                    dest="predicted_label_name", default=r'RAVLT_learning')
parser.add_argument("--prediction_fold", type=int, help="fold",
                    dest="prediction_fold", default=10)
parser.add_argument("--prediction_batch_size", type=int, help="batch size",
                    dest="prediction_batch_size", default=10)
parser.add_argument("--prediction_lr", type=float, help="learning rate",
                    dest="prediction_lr", default=3e-4)
parser.add_argument("--prediction_weight_decay_factor", type=float, help="weight decay factor",
                    dest="prediction_weight_decay_factor", default=1e-4)
parser.add_argument("--prediction_epochs", type=int, help="number of epochs",
                    dest="prediction_epochs", default=100)
parser.add_argument("--prediction_n_val_epochs", type=int, help="number of epochs",
                    dest="prediction_n_val_epochs", default=1)
parser.add_argument("--prediction_model_dir", type=str, help="models folder with prediction",
                    dest="prediction_model_dir", default=r'./model/prediction')


args = parser.parse_args()
