'''
Importing third-party libraries
'''
import os
import csv
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from monai.utils import set_determinism
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

'''
Importing internal libraries
'''
from config import args
from net.gnn.GraphNet import GraphNet
from dataloader import brain_network_dataset_diagnosis


'''
Evaluation
'''
def evaluate(net, loader):
    net.eval()
    y_labels, y_preds, y_scores = [], [], []

    with torch.no_grad():
        for _, data in loader:
            data, label = data.to(device), data.y.to(device)
            logits = net(data)
            pred_label = torch.argmax(logits, dim=-1)

            y_labels.append(label.cpu().numpy())
            y_preds.append(pred_label.cpu().numpy())
            y_scores.append(logits[:, 1].cpu().numpy())

    y_labels = np.concatenate(y_labels)
    y_preds = np.concatenate(y_preds)
    y_scores = np.concatenate(y_scores)

    accuracy = accuracy_score(y_labels, y_preds)

    tn, fp, fn, tp = confusion_matrix(y_labels, y_preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc = roc_auc_score(y_labels, y_scores)

    return accuracy, sensitivity, specificity, auc


def main():
    '''
    Log
    '''
    with open(os.path.join(args.AD_diagnosis_log_dir, 'train_log.csv'), mode='w', newline='') as f_train, \
         open(os.path.join(args.AD_diagnosis_log_dir, 'valuation_log.csv'), mode='w', newline='') as f_val, \
         open(os.path.join(args.AD_diagnosis_log_dir, 'test_log.csv'), mode='w', newline='') as f_test:

        train_log_writer = csv.writer(f_train)
        val_log_writer = csv.writer(f_val)
        test_log_writer = csv.writer(f_test)

        train_log_writer.writerow(['fold', 'epoch', 'step', 'loss'])
        val_log_writer.writerow(['fold', 'epoch', 'valuation accuracy'])
        test_log_writer.writerow(['fold', 'accuracy', 'sensitivity', 'specificity', 'auc'])

        '''
        Loading data
        '''
        dataset = brain_network_dataset_diagnosis(args.AD_diagnosis_data_load_dir,
                                                  'dataset.csv',
                                                  args.AD_diagnosis_brain_network_dir,
                                                  args.AD_diagnosis_k_num)

        ad_indices = [i for i, data_dict in enumerate(dataset.data_list) if data_dict['label'] == 0]
        cn_indices = [i for i, data_dict in enumerate(dataset.data_list) if data_dict['label'] == 1]

        ad_fold_size = len(ad_indices) // args.AD_diagnosis_fold
        cn_fold_size = len(cn_indices) // args.AD_diagnosis_fold

        accuracy = []
        sensitivity = []
        specificity = []
        auc = []

        for fold in range(args.AD_diagnosis_fold):

            if fold == args.AD_diagnosis_fold - 1:
                test_ad_indices = ad_indices[fold * ad_fold_size:]
                test_cn_indices = cn_indices[fold * cn_fold_size:]
            else:
                test_ad_start = fold * ad_fold_size
                test_ad_end = test_ad_start + ad_fold_size
                test_ad_indices = ad_indices[test_ad_start:test_ad_end]

                test_cn_start = fold * cn_fold_size
                test_cn_end = test_cn_start + cn_fold_size
                test_cn_indices = cn_indices[test_cn_start:test_cn_end]

            test_indices = test_ad_indices + test_cn_indices

            remaining_ad_indices = list(set(ad_indices) - set(test_ad_indices))
            remaining_cn_indices = list(set(cn_indices) - set(test_cn_indices))

            val_ad_fold_size = len(remaining_ad_indices) // (args.AD_diagnosis_fold - 1)
            val_cn_fold_size = len(remaining_cn_indices) // (args.AD_diagnosis_fold - 1)

            val_ad_start = (fold % (args.AD_diagnosis_fold - 1)) * val_ad_fold_size
            val_ad_end = val_ad_start + val_ad_fold_size
            val_ad_indices = remaining_ad_indices[val_ad_start:val_ad_end]

            val_cn_start = (fold % (args.AD_diagnosis_fold - 1)) * val_cn_fold_size
            val_cn_end = val_cn_start + val_cn_fold_size
            val_cn_indices = remaining_cn_indices[val_cn_start:val_cn_end]

            val_indices = val_ad_indices + val_cn_indices

            train_ad_indices = list(set(remaining_ad_indices) - set(val_ad_indices))
            train_cn_indices = list(set(remaining_cn_indices) - set(val_cn_indices))

            train_indices = train_ad_indices + train_cn_indices

            train_dataset = Subset(dataset, train_indices)
            steps_everyepoch = len(train_dataset) / args.AD_diagnosis_batch_size
            step_all = steps_everyepoch * args.AD_diagnosis_epochs
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=args.AD_diagnosis_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

            '''
            Initialize model
            '''
            net = GraphNet(input_dim=args.hidden_size, output_dim=2).to(device)
            net.train()

            '''
            Computing model flops and parameters
            '''
            # flops, params = utils.params_flops(net)
            # print('Flops: ', flops)
            # print('Params: ', params)

            '''
            Configuring optimizer and loss function
            '''
            opt = optim.AdamW(net.parameters(), lr=args.AD_diagnosis_lr, weight_decay=args.AD_diagnosis_weight_decay_factor, amsgrad=True)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=step_all)
            weights = torch.tensor([args.AD_weight, args.CN_weight])
            criterion = nn.CrossEntropyLoss(weight=weights).to(device)

            '''
            Training
            '''
            best_acc, best_epoch = 0, 0
            for epoch in range(1, args.AD_diagnosis_epochs + 1):
                for batch_idx, (_, data) in enumerate(train_loader):
                    data = data.to(device)
                    label = data.y

                    '''
                    Get predictive label
                    '''
                    output = net(data)

                    '''
                    Calculating loss
                    '''
                    loss = criterion(output, label)

                    '''
                    Printing Training Process
                    '''
                    print('Fold:[{}/{}]-----Epoch:[{}/{}]-----Batch_index:[{}/{}]-----'.format(fold + 1, args.AD_diagnosis_fold, epoch, args.AD_diagnosis_epochs, batch_idx + 1, math.ceil(steps_everyepoch)), end="")
                    print('loss: %f' % (loss.item()), flush=True, end='')
                    print('\tCurrent lr:', lr_scheduler.get_lr())
                    train_log_writer.writerow([fold + 1, epoch, batch_idx + 1, loss.item()])

                    '''
                    Gradient Zeroing and Backpropagation
                    '''
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    lr_scheduler.step()

                '''
                Valuation
                '''
                if epoch % args.AD_diagnosis_n_val_epochs == 0:
                    val_accuracy, _, _, _ = evaluate(net, val_loader)
                    print('valuation accuracy: ', val_accuracy)
                    val_log_writer.writerow([fold + 1, epoch, val_accuracy])
                    if val_accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = val_accuracy

                        save_file_name = os.path.join(args.AD_diagnosis_model_dir, 'fold' + str(fold + 1) + '_best_model.pth')
                        torch.save(net.state_dict(), save_file_name)
                        # save_optim_name = os.path.join(args.model_dir, 'optimizer.pth')
                        # torch.save(opt.state_dict(), save_optim_name)
            print('fold:', fold + 1, 'best epoch:', best_epoch, 'best accuracy:', best_acc)
            val_log_writer.writerow(['fold:', fold + 1, 'best epoch:', best_epoch, 'best accuracy:', best_acc])

            net.load_state_dict(torch.load(os.path.join(args.AD_diagnosis_model_dir, 'fold' + str(fold + 1) + '_best_model.pth')))
            print('loaded from ckpt!')

            test_accuracy, test_sensitivity, test_specificity, test_auc = evaluate(net, test_loader)
            print('fold:', fold + 1, 'test accuracy:', test_accuracy, 'test sensitivity:', test_sensitivity, 'test specificity:', test_specificity, 'test auc:', test_auc)
            test_log_writer.writerow([fold + 1, test_accuracy, test_sensitivity, test_specificity, test_auc])

            accuracy.append(test_accuracy)
            sensitivity.append(test_sensitivity)
            specificity.append(test_specificity)
            auc.append(test_auc)

        avg_accuracy = np.mean(accuracy)
        avg_sensitivity = np.mean(sensitivity)
        avg_specificity = np.mean(specificity)
        avg_auc = np.mean(auc)
        print('average accuracy:', avg_accuracy, 'average sensitivity:', avg_sensitivity, 'average specificity:', avg_specificity, 'average auc:', avg_auc)
        test_log_writer.writerow(['average accuracy:', avg_accuracy, 'average sensitivity:', avg_sensitivity, 'average specificity:', avg_specificity, 'average auc:', avg_auc])


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
