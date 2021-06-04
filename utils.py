# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os
import torch
import matplotlib
import numpy as np
import matplotlib.cm
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def colormap(image, cmap="jet"):
	image_min = torch.min(image)
	image_max = torch.max(image)
	image = (image - image_min) / (image_max - image_min)
	image = torch.squeeze(image)

	if len(image.shape) == 2:
		image = image.unsqueeze(0)

	# quantize 
	indices = torch.round(image * 255).long()
	# gather
	cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')

	colors = cm(np.arange(256))[:, :3]
	colors = torch.cuda.FloatTensor(colors)
	color_map = colors[indices].transpose(2, 3).transpose(1, 2)

	return color_map

def update_learning_rate(optimizers, scheduler):
    scheduler.step()
    lr = optimizers.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

    return lr

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def adjust_gt(gt_depth, pred_depth):
	adjusted_gt = []
	for each_depth in pred_depth:
		adjusted_gt.append(F.interpolate(gt_depth, size=[each_depth.size(2), each_depth.size(3)],
								   mode='bilinear', align_corners=True))
	return adjusted_gt


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class AverageMeter(object):
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