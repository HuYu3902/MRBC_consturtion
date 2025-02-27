#!/usr/bin/env bash
import os
import SimpleITK as sitk

'''
N4 bias field correction
'''
def N4BiasFieldCorrection(data_dir, result_dir):
    os.makedirs(os.path.join(result_dir, 'step1_N4'), exist_ok=True)
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i])
        output_file = os.path.join(result_dir, 'step1_N4', files_list[i])
        cmd_N4 = 'N4BiasFieldCorrection -d 3 -i ' + input_file + ' -o ' + output_file
        print(cmd_N4)
        os.system(cmd_N4)
        print('N4 bias field correction completed for ' + files_list[i])

'''
Affine alignment to MNI 152 template
'''
def Affine(result_dir, MNI152_template = r'MNI152_template/MNI152.nii'):
    data_dir = os.path.join(result_dir, 'step1_N4')
    os.makedirs(os.path.join(result_dir, 'step2_affine', 'intermediate_results'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'step2_affine', 'final_results'), exist_ok=True)
    intermediate_result_dir = os.path.join(result_dir, 'step2_affine', 'intermediate_results')
    final_result_dir = os.path.join(result_dir, 'step2_affine', 'final_results')
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i])
        subject_ID = files_list[i].split('.')[0]
        os.makedirs(os.path.join(intermediate_result_dir, subject_ID), exist_ok=True)
        regImage = os.path.join(intermediate_result_dir, subject_ID, 'Output.nii.gz')

        cmd_ANTs = 'ANTS 3 -m CC[' + MNI152_template + ', ' + input_file + ',1,2] -o ' + regImage + ' -i 0'
        print(cmd_ANTs)
        os.system(cmd_ANTs)

        Affine_dir = os.path.join(intermediate_result_dir, subject_ID, 'OutputAffine.txt')
        output_file = os.path.join(final_result_dir, subject_ID + '.nii.gz')
        cmd_MNI152 = 'WarpImageMultiTransform 3 ' + input_file + ' ' + output_file + ' ' + Affine_dir + ' -R ' + MNI152_template
        print(cmd_MNI152)
        os.system(cmd_MNI152)

        print('Affine alignment completed for ' + files_list[i])

'''
Nonlinearly registering MNI 152
'''
def SyN(result_dir, MNI152_template = r'MNI152_template/MNI152.nii'):
    data_dir = os.path.join(result_dir, 'step2_affine', 'final_results')
    os.makedirs(os.path.join(result_dir, 'step3_SyN', 'intermediate_results'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'step3_SyN', 'final_results'), exist_ok=True)
    intermediate_result_dir = os.path.join(result_dir, 'step3_SyN', 'intermediate_results')
    final_result_dir = os.path.join(result_dir, 'step3_SyN', 'final_results')
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i])
        subject_ID = files_list[i].split(".")[0]
        os.makedirs(os.path.join(intermediate_result_dir, subject_ID), exist_ok=True)
        regImage = os.path.join(intermediate_result_dir, subject_ID, 'output.nii.gz')

        cmd_ANTs = 'ANTS 3 -m CC[' + MNI152_template + ',' + input_file + ',1,2] -t SyN[0.25] -r Gauss[3,1.0] -o ' + regImage + ' -i 201x201x201 --number-of-affine-iterations 100x100x100'
        print(cmd_ANTs)
        os.system(cmd_ANTs)

        Affine_file = os.path.join(intermediate_result_dir, subject_ID, 'outputAffine.txt')
        Warp_file = os.path.join(intermediate_result_dir, subject_ID, 'outputWarp.nii.gz')
        output_file = os.path.join(final_result_dir, subject_ID + '.nii.gz')
        cmd_SyN = 'WarpImageMultiTransform 3 ' + input_file + ' ' + output_file + ' ' + Warp_file + ' ' + Affine_file + ' -R ' + MNI152_template
        print(cmd_SyN)
        os.system(cmd_SyN)

        print('Nonlinear registration completed for ' + files_list[i])

'''
Skull stripping
'''
def SkullStripping(result_dir, T_template0 = r'skull_stripping_template/T_template0.nii.gz', ProbabilityMask = r'skull_stripping_template/T_template0_BrainCerebellumProbabilityMask.nii.gz', RegistrationMask = r'skull_stripping_template/T_template0_BrainCerebellumRegistrationMask.nii.gz'):
    data_dir = os.path.join(result_dir, 'step3_SyN', 'final_results')
    os.makedirs(os.path.join(result_dir, 'step4_skull_stripping'), exist_ok=True)
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i])
        subject_ID = files_list[i].split(".")[0]
        os.makedirs(os.path.join(result_dir, 'step4_skull_stripping', subject_ID), exist_ok=True)
        output_file = os.path.join(result_dir, 'step4_skull_stripping', subject_ID + "/")
        cmd_Skull_Stripping = 'antsBrainExtraction.sh -d 3 -a ' + input_file + ' -e ' + T_template0 + ' -m ' + ProbabilityMask + ' -f ' + RegistrationMask + ' -o ' + output_file
        print(cmd_Skull_Stripping)
        os.system(cmd_Skull_Stripping)

        print('Skull stripping completed for ' + files_list[i])

'''
Crop
'''
def Crop_Subfun(input_file, output_file):
    # Read NIfTI image
    image = sitk.ReadImage(input_file)

    # Get the size of the original image
    orig_size = image.GetSize()

    # Calculate the starting indices for cropping to center the brain and maintain a size of 160x192x160
    # It is assumed here that the image is sufficiently large in each dimension to crop the desired region
    crop_size = [160, 192, 160]
    crop_start = [(dim - new_dim) // 2 for dim, new_dim in zip(orig_size, crop_size)]

    # Create cropping indices
    crop_index = [
        slice(crop_start[0], crop_start[0] + crop_size[0]),
        slice(crop_start[1], crop_start[1] + crop_size[1]),
        slice(crop_start[2], crop_start[2] + crop_size[2])
    ]

    # Perform cropping using the indices
    cropped_image = image[crop_index]

    cropped_image.SetOrigin(image.GetOrigin())
    cropped_image.SetDirection(image.GetDirection())
    cropped_image.SetSpacing(image.GetSpacing())

    # Save the cropped NIfTI image
    sitk.WriteImage(cropped_image, output_file)

def Crop(result_dir):
    data_dir = os.path.join(result_dir, 'step4_skull_stripping')
    os.makedirs(os.path.join(result_dir, 'step5_crop'), exist_ok=True)
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i], 'BrainExtractionBrain.nii.gz')
        output_file = os.path.join(result_dir, 'step5_crop', files_list[i] + '.nii.gz')

        Crop_Subfun(input_file, output_file)

        print('Crop completed for ' + files_list[i])

'''
Normalization
'''
def Norm_Subfunction(img_array):
    img_array_max = img_array.max()
    img_array_min = img_array.min()
    img_array = (img_array - img_array_min) / (img_array_max - img_array_min)
    return img_array

def Norm(result_dir):
    data_dir = os.path.join(result_dir, 'step5_crop')
    os.makedirs(os.path.join(result_dir, 'step6_norm'), exist_ok=True)
    files_list = os.listdir(data_dir)
    for i in range(len(files_list)):
        input_file = os.path.join(data_dir, files_list[i])
        output_file = os.path.join(result_dir, 'step6_norm', files_list[i])

        img = sitk.ReadImage(input_file)
        img_array = sitk.GetArrayFromImage(img)

        img_norm_array = Norm_Subfunction(img_array)

        img_norm = sitk.GetImageFromArray(img_norm_array)
        img_norm.SetOrigin(img.GetOrigin())
        img_norm.SetDirection(img.GetDirection())
        img_norm.SetSpacing(img.GetSpacing())

        sitk.WriteImage(img_norm, output_file)

        print('Normalization completed for ' + files_list[i])
