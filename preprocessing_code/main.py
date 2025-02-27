#!/usr/bin/env bash
import os
import shutil
from utils import N4BiasFieldCorrection, Affine, SyN, SkullStripping, Crop, Norm


def main(data_dir, intermediate_result_dir, final_result_dir):
    '''
    Creating result directory
    '''
    os.makedirs(intermediate_result_dir, exist_ok=True)
    os.makedirs(final_result_dir, exist_ok=True)

    '''
    Step 1: Performing N4 bias field correction
    '''
    N4BiasFieldCorrection(data_dir, intermediate_result_dir)

    '''
    Step 2: Performing affine alignment
    '''
    Affine(intermediate_result_dir)

    '''
    Step 3: Performing nonlinear registration using the SyN algorithm in ANTs
    '''
    SyN(intermediate_result_dir)

    '''
    Step 4: Performing skull stripping
    '''
    SkullStripping(intermediate_result_dir)

    '''
    Step 5: Performing crop
    '''
    Crop(intermediate_result_dir)

    '''
    Step 6: Performing Normalization
    '''
    Norm(intermediate_result_dir)

    '''
    Copy the final results to the final results folder
    '''
    img_dir = os.path.join(intermediate_result_dir, 'step6_norm')
    img_list = os.listdir(img_dir)
    for i in range(len(img_list)):
        source_file = os.path.join(img_dir, img_list[i])
        destination_file = os.path.join(final_result_dir, img_list[i])
        shutil.copy(source_file, destination_file)
        print('Copy completed for ' + img_list[i])

    shutil.rmtree(intermediate_result_dir)


if __name__ == '__main__':
    '''
    Defining paths for unprocessed and processed results
    '''
    data_dir = r"../unpreprocessed/CoRR/Utah2/session_1"
    intermediate_result_dir = r"../CoRR/Utah2/session_1/intermediate_result"
    final_result_dir = r"../CoRR/Utah2/session_1"

    main(data_dir, intermediate_result_dir, final_result_dir)
