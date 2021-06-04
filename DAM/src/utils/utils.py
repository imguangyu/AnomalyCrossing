import csv
import numpy as np
import torch
import time
import shutil
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
# from sklearn.metrics import confusion_matrix


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu_id = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu_id)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)

class Timer(object):
	"""
	docstring for Timer
	"""
	def __init__(self):
		super(Timer, self).__init__()
		self.total_time = 0.0
		self.calls = 0
		self.start_time = 0.0
		self.diff = 0.0
		self.average_time = 0.0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average = False):
		self.diff = time.time() - self.start_time
		self.calls += 1
		self.total_time += self.diff
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff

	def format(self, time):
		m,s = divmod(time, 60)
		h,m = divmod(m, 60)
		d,h = divmod(h, 24)
		return ("{}d:{}h:{}m:{}s".format(int(d), int(h), int(m), int(s)))

	def end_time(self, extra_time):
		"""
		calculate the end time for training, show local time
		"""
		localtime= time.asctime(time.localtime(time.time() + extra_time))
		return localtime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


class TrainingHelper(object):
    def __init__(self, image):
        self.image = image
    def congratulation(self):
        """
        if finish training success, print congratulation information
        """
        for i in range(40):
            print('*')*i
            print('finish training')


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output: 16(batch_size) x 101
    target: 16 x 1
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 5 x 16
    #print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_mixup(output, targets, target_a, target_b, lam, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() #5 x 20
    correct_1 = pred.eq(target_a.data.view(1, -1).expand_as(pred))
    correct_2 = pred.eq(target_b.data.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = lam * correct_1[:k].view(-1).float().sum(0) + (1-lam) * correct_2[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
