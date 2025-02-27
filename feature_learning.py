'''
Importing third-party libraries
'''
import os
import csv
import math
import time
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from monai.utils import set_determinism

'''
Importing internal libraries
'''
import utils
from losses import NCC
from config import args
from dataloader import brian_image_dataset
from net.rec.maskConvNet import maskConvAutoEncoder


def main():
    '''
    Log
    '''
    with open(os.path.join(args.feature_learning_log_dir, 'log.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'total loss', 'mse loss', 'ncc loss'])

        '''
        Initialize model
        '''
        net = maskConvAutoEncoder(hidden_size=args.hidden_size).to(device)
        net.train()

        '''
        Computing model flops and parameters
        '''
        # flops, params = utils.params_flops(net)
        # print('Flops: ', flops)
        # print('Params: ', params)

        '''
        Loading data
        '''
        dataset = brian_image_dataset(data_load_dir=args.feature_learning_data_load_dir,
                                      data_load_csv='feature_learning_data.csv',
                                      data_dir=args.feature_learning_data_dir)
        loader = DataLoader(dataset, batch_size=args.feature_learning_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        number = len(dataset)
        steps_everyepoch = number/args.feature_learning_batch_size
        step_all = steps_everyepoch * args.feature_learning_epochs

        '''
        Configuring optimizer and loss
        '''
        opt = optim.AdamW(net.parameters(), lr=args.feature_learning_lr, weight_decay=args.feature_learning_weight_decay_factor, amsgrad=True)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=step_all)
        mse_loss_fn = nn.MSELoss()
        ncc_loss_fn = NCC()

        '''
        Training
        '''
        for epoch in range(1, args.feature_learning_epochs + 1):
            for batch_idx, (_, image, mask) in enumerate(loader):
                image = image.to(device).float()
                mask = mask.to(device).float()
                masked_image = image * mask

                '''
                Get reconstructed images
                '''
                x_rec = net(masked_image)

                '''
                Calculating loss
                '''
                mse_loss = mse_loss_fn(image, x_rec)
                ncc_loss = ncc_loss_fn(image, x_rec)
                loss = mse_loss + ncc_loss

                '''
                Printing Training Process
                '''
                print('Epoch:[{}/{}]-----Batch_index:[{}/{}]-----'.format(epoch, args.feature_learning_epochs, batch_idx + 1, math.ceil(steps_everyepoch)), end="")
                print('total loss: %f   mse loss: %f   ncc loss: %f' % (loss.item(), mse_loss.item(), ncc_loss.item()), flush=True, end='')
                print('\tCurrent lr:', lr_scheduler.get_lr())
                writer.writerow([epoch, batch_idx + 1, loss.item(), mse_loss.item(), ncc_loss.item()])

                '''
                Backpropagation
                '''
                opt.zero_grad()
                loss.backward()
                opt.step()
                lr_scheduler.step()

            '''
            Saving checkpoint
            '''
            save_file_name = os.path.join(args.feature_learning_model_dir, 'model_' + str(epoch) + '.pth')
            torch.save(net.state_dict(), save_file_name)
            # save_optim_name = os.path.join(args.model_dir, 'optimizer_' + str(epoch) + '.pth')
            # torch.save(opt.state_dict(), save_optim_name)


if __name__ == "__main__":
    '''
    Start time
    '''
    start_time = time.time()

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

    '''
    End time
    '''
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))