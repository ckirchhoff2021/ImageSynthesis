from src.train import Trainer
from dataloader.dataloader import get_loader
import os
from config.config import get_config

from data import corruptFolder_dataloader, corrupt_dataloader


def main(config):
    # make directory not existed
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    print(f"ESRGAN start")


    '''
    data_folder = '/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen'
    train_count = 1e8
    data_loader = corruptFolder_dataloader(data_folder, train_count, 4, num_workers=4, size=256, distributed=False)
    '''

    data_dir = '/data/juicefs_hz_cv_v3/11145199/datas/'
    train_file ='hq_train.json'

    config.checkpoint_dir = '/data/juicefs_hz_cv_v2/11145199/deblur/results/esrganHQ'
    config.sample_dir = '/data/juicefs_hz_cv_v2/11145199/deblur/results/esrganHQ/samples'
    data_loader = corrupt_dataloader(data_dir, train_file, 8000, 2, 256, num_workers=4, distributed=False)

    trainer = Trainer(config, data_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
