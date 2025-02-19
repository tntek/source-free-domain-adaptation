import os
import numpy as np
import torch
import random
import time
import src.methods.oh.nrc as NRC
import src.methods.oh.shot as SHOT
import src.methods.oh.sclm as SCLM
import src.methods.oh.cowa as COWA
import src.methods.oh.gkd as GKD
import src.methods.oh.tpds as TPDS
import src.methods.oh.nrc_vs as Nrc_vs
import src.methods.oh.lcfd as LCFD
import src.methods.oh.difo as DIFO
import src.methods.oh.tsd as TSD
import src.methods.oh.difo_50 as DIFO_50

import src.methods.oh.plue as PLUE
import src.methods.oh.adacontrast as ADACONTRAST
import src.methods.oh.source as SOURCE
from conf import cfg, load_cfg_from_args


if __name__ == "__main__":
    load_cfg_from_args()
    print("+++++++++++++++++++IB+++++++++++++++++++++")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    cfg.type = cfg.domain
    cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.s_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S] + '_list.txt'
    cfg.savename = cfg.MODEL.METHOD
    start =time.time()
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    random.seed(cfg.SETTING.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    if cfg.SETTING.DATASET == 'office-home':
        if cfg.DA == 'pda':
            cfg.class_num = 65
            cfg.src_classes = [i for i in range(65)]
            cfg.tar_classes = [i for i in range(25)]

    if cfg.MODEL.METHOD == "nrc":
        print("using nrc method")
        acc = NRC.train_target(cfg)

    elif cfg.MODEL.METHOD == "shot":
        print("using shot method")
        acc = SHOT.train_target(cfg)

    elif cfg.MODEL.METHOD == "sclm":
        print("using sclm method")
        acc = SCLM.train_target(cfg)

    elif cfg.MODEL.METHOD == "cowa":
        print("using cowa method")
        cfg.prefix = '{}_alpha{}_lr{}_epoch{}_interval{}_seed{}_warm{}'.format(
            cfg.COWA.COEFF, cfg.COWA.ALPHA, cfg.OPTIM.LR, cfg.TEST.MAX_EPOCH, cfg.TEST.INTERVAL, cfg.SETTING.SEED, cfg.COWA.WARM
        )
        acc = COWA.train_target(cfg)

    elif cfg.MODEL.METHOD == "gkd":
        print("using gkd method")
        acc = GKD.train_target(cfg)

    elif cfg.MODEL.METHOD == "tpds":
        print("using tpds method")
        acc = TPDS.train_target(cfg)

    elif cfg.MODEL.METHOD == "nrc_vs":
        print("using nrc method")
        acc = Nrc_vs.train_target(cfg)

    elif cfg.MODEL.METHOD == "lcfd":
        print("using lcfd method")
        acc = LCFD.train_target(cfg)

    elif cfg.MODEL.METHOD == "difo":
        print("using difo method")
        if 'RN' in cfg.DIFO.ARCH:
            acc = DIFO_50.train_target(cfg)
        else:
            acc = DIFO.train_target(cfg)

    elif cfg.MODEL.METHOD == "tsd":
        print("using plue method")
        acc = TSD.train_target(cfg)

    elif cfg.MODEL.METHOD == "plue":
        print("using plue method")
        acc = PLUE.train_target(cfg)

    elif cfg.MODEL.METHOD == "adacontrast":
        print("using adacontrast method")
        acc = ADACONTRAST.train_target(cfg)

    elif cfg.MODEL.METHOD == "source":
        print("training source model")
        acc = SOURCE.train_source(cfg)

    end = time.time()
    all_time = 'Running time: %s Seconds'%(round(end-start, 2))
    print(all_time)
