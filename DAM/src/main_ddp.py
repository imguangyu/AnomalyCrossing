from option import args
import datetime
from pt import pretext_train
from ft import fine_tune_train_and_val
from pt_dota_ddp import pretext_train_dota
import os
from utils.recoder import Record
import utils.utils as utils


def main():
    utils.init_distributed_mode(args)
    args.date = datetime.datetime.today().strftime('%m-%d-%H%M')
    recorder = None
    if utils.is_main_process():
        recorder = Record(args)
    message = ""
    if args.method == 'dota':
        args.status = 'pt'
        checkpoints_path = pretext_train_dota(args, recorder)
        # print("finished dota , the weight is in: {}".format(args.ft_weights))
    else:
        Exception("wrong method!")

if __name__ == '__main__':
    main()
