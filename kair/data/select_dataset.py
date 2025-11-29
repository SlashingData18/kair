

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['l', 'low-quality', 'input-only']:
        from kair.data.dataset_l import DatasetL as D

    # -----------------------------------------
    # denoising
    # -----------------------------------------
    elif dataset_type in ['dncnn', 'denoising']:
        from kair.data.dataset_dncnn import DatasetDnCNN as D

    elif dataset_type in ['dnpatch']:
        from kair.data.dataset_dnpatch import DatasetDnPatch as D

    elif dataset_type in ['ffdnet', 'denoising-noiselevel']:
        from kair.data.dataset_ffdnet import DatasetFFDNet as D

    elif dataset_type in ['fdncnn', 'denoising-noiselevelmap']:
        from kair.data.dataset_fdncnn import DatasetFDnCNN as D

    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    elif dataset_type in ['sr', 'super-resolution']:
        from kair.data.dataset_sr import DatasetSR as D

    elif dataset_type in ['srmd']:
        from kair.data.dataset_srmd import DatasetSRMD as D

    elif dataset_type in ['dpsr', 'dnsr']:
        from kair.data.dataset_dpsr import DatasetDPSR as D

    elif dataset_type in ['usrnet', 'usrgan']:
        from kair.data.dataset_usrnet import DatasetUSRNet as D

    elif dataset_type in ['bsrnet', 'bsrgan', 'blindsr']:
        from kair.data.dataset_blindsr import DatasetBlindSR as D

    # -------------------------------------------------
    # JPEG compression artifact reduction (deblocking)
    # -------------------------------------------------
    elif dataset_type in ['jpeg']:
        from kair.data.dataset_jpeg import DatasetJPEG as D

    # -----------------------------------------
    # video restoration
    # -----------------------------------------
    elif dataset_type in ['videorecurrenttraindataset']:
        from kair.data.dataset_video_train import VideoRecurrentTrainDataset as D
    elif dataset_type in ['videorecurrenttrainnonblinddenoisingdataset']:
        from kair.data.dataset_video_train import VideoRecurrentTrainNonblindDenoisingDataset as D
    elif dataset_type in ['videorecurrenttrainvimeodataset']:
        from kair.data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    elif dataset_type in ['videorecurrenttrainvimeovfidataset']:
        from kair.data.dataset_video_train import VideoRecurrentTrainVimeoVFIDataset as D
    elif dataset_type in ['videorecurrenttestdataset']:
        from kair.data.dataset_video_test import VideoRecurrentTestDataset as D
    elif dataset_type in ['singlevideorecurrenttestdataset']:
        from kair.data.dataset_video_test import SingleVideoRecurrentTestDataset as D
    elif dataset_type in ['videotestvimeo90kdataset']:
        from kair.data.dataset_video_test import VideoTestVimeo90KDataset as D
    elif dataset_type in ['vfi_davis']:
        from kair.data.dataset_video_test import VFI_DAVIS as D
    elif dataset_type in ['vfi_ucf101']:
        from kair.data.dataset_video_test import VFI_UCF101 as D
    elif dataset_type in ['vfi_vid4']:
        from kair.data.dataset_video_test import VFI_Vid4 as D


    # -----------------------------------------
    # common
    # -----------------------------------------
    elif dataset_type in ['plain']:
        from kair.data.dataset_plain import DatasetPlain as D

    elif dataset_type in ['plainpatch']:
        from kair.data.dataset_plainpatch import DatasetPlainPatch as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
