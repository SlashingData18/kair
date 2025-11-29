import os
import argparse
import math
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kair.utils import utils_logger
from kair.utils import utils_image as util
from kair.utils import utils_option as option
from kair.utils.utils_dist import get_dist_info, init_dist

from kair.data.select_dataset import define_Dataset
from kair.models.select_model import define_Model


def _find_and_set_checkpoints(opt, net_types):
    # Look for last checkpoints for provided net_types and store in opt['path'] entries
    init_iters = {}
    for nt in net_types:
        init_iter, init_path = option.find_last_checkpoint(opt['path']['models'], net_type=nt)
        key = f'pretrained_net{nt}' if nt.startswith(('G','D','E')) else f'pretrained_{nt}'
        # some callers expect keys like pretrained_netG or pretrained_optimizerG - set the explicit ones
        if nt.lower().startswith('optimizer'):
            # e.g. optimizerG -> pretrained_optimizerG
            key = f'pretrained_{nt}'
        opt['path'][f'pretrained_{nt}'] = init_path
        # Backwards compatibility for netX keys
        if nt in ('G','D','E'):
            opt['path'][f'pretrained_net{nt}'] = init_path
        init_iters[nt] = init_iter

    if init_iters:
        current_step = max(init_iters.values())
    else:
        current_step = 0
    return current_step


def run(json_path, net_checkpoint_types=None, dist=False, local_rank=0, save_images=False, epochs=100000):
    """Generic training runner used by modular main_train_* wrappers.

    Args:
        json_path (str): path to option json file.
        net_checkpoint_types (list[str]): list of net_type strings to search for checkpoints, e.g. ['G'] or ['G','D','E','optimizerG']
        dist (bool): whether to use distributed training.
        local_rank (int): local rank for distributed training.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=local_rank)
    parser.add_argument('--dist', default=dist)
    parser.add_argument('--save-images', dest='save_images', action='store_true', help='Save test images during validation / test.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    # allow runner param to override opt-dist if provided
    opt['dist'] = parser.parse_args().dist

    # distributed settings
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # update opt: find latest checkpoints according to provided net types
    if net_checkpoint_types is None:
        net_checkpoint_types = ['G']
    current_step = _find_and_set_checkpoints(opt, net_checkpoint_types)

    # set a default border value used by some training scripts
    border = opt.get('border', None)
    if border is None:
        border = opt.get('scale', 0)

    # save opt to disk (only rank 0)
    if opt['rank'] == 0:
        option.save(opt)

    # return None for missing key convenience
    opt = option.dict_to_nonedict(opt)

    # configure logger (only rank 0)
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    else:
        logger = logging.getLogger('train')

    # seed
    seed = opt['train'].get('manual_seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
    if opt['rank'] == 0:
        logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # if CLI args provided a save-images flag, use it (CLI takes precedence), otherwise use provided param or option
    parsed_args = parser.parse_args()
    # if the flag was passed explicitly on CLI, it will be True; otherwise keep function param value
    cli_save_images = getattr(parsed_args, 'save_images', False)
    save_images = bool(cli_save_images) or bool(save_images)

    # save opt to disk (only rank 0)
    dataset_type = None
    train_loader = None
    test_loader = None
    train_sampler = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_type = dataset_opt.get('dataset_type', None)
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))

            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=max(1, dataset_opt['dataloader_batch_size'] // max(1, opt.get('num_gpu', 1))),
                                          shuffle=False,
                                          num_workers=max(1, dataset_opt['dataloader_num_workers'] // max(1, opt.get('num_gpu', 1))),
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # initialize model
    model = define_Model(opt)
    model.init_train()

    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # main training loop
    for epoch in range(epochs):
        # distributed: set epoch on sampler
        if opt['dist'] and train_sampler is not None:
            # some scripts use epoch, others epoch + seed
            try:
                train_sampler.set_epoch(epoch + seed)
            except Exception:
                train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # update learning rate
            model.update_learning_rate(current_step)

            # feed and optimize
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # merge batchnorm special handling
            if opt.get('merge_bn', False) and opt.get('merge_bn_startpoint', None) == current_step:
                if opt['rank'] == 0:
                    logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                if opt['rank'] == 0:
                    model.print_network()

            # logging
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # save
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # testing
            if current_step % opt['train']['checkpoint_test'] == 0 and test_loader is not None and opt['rank'] == 0:
                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    # test_data might have different path keys. Guess L_path by common pattern
                    image_name_ext = os.path.basename(test_data.get('L_path', [test_data.get('H_path', ['unknown'])])[0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    # only create and save images when asked; otherwise skip writing to disk
                    if save_images:
                        util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    if save_images:
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imsave(E_img, save_img_path)

                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

    if opt['rank'] == 0:
        logger.info('Saving the final model.')
    model.save('latest')
    if opt['rank'] == 0:
        logger.info('End of training.')

    # return the trained model for programmatic use
    return model


if __name__ == '__main__':
    # simple CLI for running generic runner
    run('options/train_dncnn.json')
