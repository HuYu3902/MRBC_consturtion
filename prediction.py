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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

'''
Importing internal libraries
'''
from config import args
from net.gnn.GraphNet import GraphNet
from dataloader import brain_network_dataset_prediction


'''
Evaluation
'''
def evaluate(net, loader):
    net.eval()
    y_labels, y_preds = [], []

    with torch.no_grad():
        for _, data in loader:
            data, label = data.to(device), data.y.to(device)
            logits = net(data)

            y_labels.append(label.cpu().numpy())
            y_preds.append(logits.view(-1).cpu().numpy())

    y_labels = np.concatenate(y_labels)
    y_preds = np.concatenate(y_preds)

    mae = mean_absolute_error(y_labels, y_preds)
    rmse = np.sqrt(mean_squared_error(y_labels, y_preds))
    r_value, p_value = pearsonr(y_labels, y_preds)

    return mae, rmse, r_value, p_value


def main():
    '''
    Log
    '''
    with open(os.path.join(args.prediction_log_dir, args.predicted_label_name + '_train_log.csv'), mode='w', newline='') as f_train, \
         open(os.path.join(args.prediction_log_dir, args.predicted_label_name + '_valuation_log.csv'), mode='w', newline='') as f_val, \
         open(os.path.join(args.prediction_log_dir, args.predicted_label_name + '_test_log.csv'), mode='w', newline='') as f_test:

        train_log_writer = csv.writer(f_train)
        val_log_writer = csv.writer(f_val)
        test_log_writer = csv.writer(f_test)

        train_log_writer.writerow(['fold', 'epoch', 'step', 'loss'])
        val_log_writer.writerow(['fold', 'epoch', 'MAE', 'RMSE', 'Pearson R'])
        test_log_writer.writerow(['fold', 'MAE', 'RMSE', 'Pearson R', 'Pearson P'])

        '''
        Loading data
        '''
        dataset = brain_network_dataset_prediction(args.prediction_data_load_dir,
                                                   args.predicted_label_name + '_dataset.csv',
                                                   args.prediction_brain_network_dir,
                                                   args.prediction_k_num,
                                                   args.phenotypic_information_csv,
                                                   args.predicted_label_name)

        indices = [i for i, data_dict in enumerate(dataset.data_list)]

        fold_size = len(indices) // args.prediction_fold

        mae = []
        rmse = []
        r_values = []
        p_values = []

        for fold in range(args.prediction_fold):

            if fold == args.prediction_fold - 1:
                test_indices = indices[fold * fold_size:]
            else:
                test_start = fold * fold_size
                test_end = test_start + fold_size
                test_indices = indices[test_start:test_end]

            remaining_indices = list(set(indices) - set(test_indices))

            val_fold_size = len(remaining_indices) // (args.prediction_fold - 1)

            val_start = (fold % (args.prediction_fold - 1)) * val_fold_size
            val_end = val_start + val_fold_size
            val_indices = remaining_indices[val_start:val_end]

            train_indices = list(set(remaining_indices) - set(val_indices))

            train_dataset = Subset(dataset, train_indices)
            steps_everyepoch = len(train_dataset) / args.prediction_batch_size
            step_all = steps_everyepoch * args.prediction_epochs
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=args.prediction_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)

            '''
            Initialize model
            '''
            net = GraphNet(input_dim=args.hidden_size, output_dim=1).to(device)
            net.train()

            '''
            Configuring optimizer and loss function
            '''
            opt = optim.AdamW(net.parameters(), lr=args.prediction_lr, weight_decay=args.prediction_weight_decay_factor, amsgrad=True)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
            criterion = nn.L1Loss().to(device)

            '''
            Training
            '''
            best_mae, best_epoch = float('inf'), 0
            for epoch in range(1, args.prediction_epochs + 1):
                for batch_idx, (_, data) in enumerate(train_loader):
                    data = data.to(device)
                    label = data.y

                    '''
                    Get predicted label
                    '''
                    output = net(data)

                    '''
                    Calculating loss
                    '''
                    loss = criterion(output.view(-1), label)

                    '''
                    Printing Training Process
                    '''
                    print(f'Fold: [{fold + 1}/{args.prediction_fold}]-----Epoch: [{epoch}/{args.prediction_epochs}]-----Batch_index: [{batch_idx + 1}/{math.ceil(steps_everyepoch)}]-----loss: {loss.item()}')
                    train_log_writer.writerow([fold + 1, epoch, batch_idx + 1, loss.item()])

                    '''
                    Gradient Zeroing and Backpropagation
                    '''
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                '''
                Valuation
                '''
                if epoch % args.prediction_n_val_epochs == 0:
                    val_mae, val_rmse, val_r, val_p = evaluate(net, val_loader)
                    print(f'valuation MAE: {val_mae} valuation RMSE: {val_rmse} valuation Pearson R: {val_r}')
                    val_log_writer.writerow([fold + 1, epoch, val_mae, val_rmse, val_r])
                    if val_mae < best_mae:
                        best_epoch = epoch
                        best_mae = val_mae

                        save_file_name = os.path.join(args.prediction_model_dir, args.predicted_label_name + f'_fold{fold + 1}_best_model.pth')
                        torch.save(net.state_dict(), save_file_name)
                    lr_scheduler.step(val_mae)

            print(f'fold: {fold + 1} best epoch: {best_epoch} best MAE: {best_mae}')
            val_log_writer.writerow([f'fold: {fold + 1}', f'best epoch: {best_epoch}', f'best MAE: {best_mae}'])

            net.load_state_dict(torch.load(os.path.join(args.prediction_model_dir, args.predicted_label_name + f'_fold{fold + 1}_best_model.pth')))
            print('loaded from ckpt!')

            test_mae, test_rmse, test_r, test_p = evaluate(net, test_loader)
            print(f'fold: {fold + 1} test MAE: {test_mae} test RMSE: {test_rmse} test Pearson R: {test_r}, P-value: {test_p}')
            test_log_writer.writerow([fold + 1, test_mae, test_rmse, test_r, test_p])

            mae.append(test_mae)
            rmse.append(test_rmse)
            r_values.append(test_r)
            p_values.append(test_p)

        avg_mae = np.mean(mae)
        std_mae = np.std(mae)
        avg_rmse = np.mean(rmse)
        std_rmse = np.std(rmse)
        avg_r = np.mean(r_values)
        std_r = np.std(r_values)
        avg_p = np.mean(p_values)
        std_p = np.std(p_values)

        print(f'average MAE: {avg_mae} ± {std_mae}')
        print(f'average RMSE: {avg_rmse} ± {std_rmse}')
        print(f'average Pearson R: {avg_r} ± {std_r}')
        print(f'average P-value: {avg_p} ± {std_p}')

        test_log_writer.writerow([
            f'average MAE: {avg_mae} ± {std_mae}',
            f'average RMSE: {avg_rmse} ± {std_rmse}',
            f'average Pearson R: {avg_r} ± {std_r}',
            f'average P-value: {avg_p} ± {std_p}'
        ])


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
