import os
import numpy as np
import torch
import random
import time
import src.methods.net.nrc as NRC
import src.methods.net.shot as SHOT
import src.methods.net.sclm as SCLM
import src.methods.net.cowa as COWA
import src.methods.net.gkd as GKD
import src.methods.net.tpds as TPDS
import src.methods.net.CausalDA as CausalDA
import src.methods.net.difo as DIFO
import src.methods.net.tsd as TSD
import src.methods.net.ProDe as ProDe
import src.methods.net.plue as PLUE
import src.methods.net.adacontrast as ADACONTRAST
import src.methods.net.source as SOURCE
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

    elif cfg.MODEL.METHOD == "CausalDA":
        print("using CausalDA method")
        acc = CausalDA.train_target(cfg)

    elif cfg.MODEL.METHOD == "difo":
        print("using difo method")
        acc = DIFO.train_target(cfg)

    elif cfg.MODEL.METHOD == "tsd":
        print("using tsd method")
        acc = TSD.train_target(cfg)

    elif cfg.MODEL.METHOD == "ProDe":
        print("using ProDe method")
        acc = ProDe.train_target(cfg)

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
