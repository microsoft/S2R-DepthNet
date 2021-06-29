# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os 
import torch
import argparse
import loaddata
import matplotlib
import numpy as np
from utils import *
import matplotlib.cm
import torch.nn as nn
import DSAModules

import torch.nn.parallel
import matplotlib as mpl
from models import modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

plt.switch_backend('agg')
plt.set_cmap("jet")

## =========================== Parameters =================
parser = argparse.ArgumentParser(description="S2R-DepthNet-Test.")
parser.add_argument('--dataset', type=str, default='VKITTI', help='synthetic domain')                       # *******
parser.add_argument('--root',  type=str, default='', help='path to source dataset.')                        # *******
parser.add_argument('--test_datafile', type=str, default='', help='stores data list, in syn_root')         # *******
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')                                # *******
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')                       # *******
parser.add_argument('--loadSize', nargs='+', type=int, default=286, help='scale images to this size')           # *******
parser.add_argument('--out_dir', type=str, default="out", help="the path of log")
parser.add_argument('--Shared_Struct_Encoder_path', type=str, default="", help='the path of Shared_Struct_Encoder models')
parser.add_argument('--Struct_Decoder_path', type=str, default="", help='the path of Struct_Decoder models')
parser.add_argument('--DepthNet_path', type=str, default="", help='the path of DepthNet models')
parser.add_argument('--DSAModle_path', type=str, default='', help='the path of DSAModle')


def save_test(handle, result1_log):
	'''
	This function save the test metrics in a given file.
	------
	handle: text file handle
	result_log: the metrics results, a 2D list
	'''
	abs_rel_1 = np.array(result1_log[0]).mean()
	sq_rel_1 = np.array(result1_log[1]).mean()
	rmse_1 = np.array(result1_log[2]).mean()
	rmse_log_1 = np.array(result1_log[3]).mean()
	a1_1 = np.array(result1_log[4]).mean()
	a2_1 = np.array(result1_log[5]).mean()
	a3_1 = np.array(result1_log[6]).mean()


	# write test result to test file by using handle
	handle.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n" \
			.format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))

	handle.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n"\
			.format(abs_rel_1, sq_rel_1, rmse_1, rmse_log_1, a1_1, a2_1, a3_1))



def kitti_metrics_preprocess(pred, gt):
	'''
	This function do some preprocessing before metrics calculation
	- check zero value to avoid numerical problems;
	-
	Note that the input 'pred' and 'gt' are both 4D nparrays
	return the corresponding image pair 
	'''
	# squeeze the first and last idx(which is one in test processing)

	pred = np.squeeze(pred)
	gt = np.squeeze(gt)

	min_depth = 1e-3
	max_depth = 80
	pred[pred < min_depth] = min_depth
	pred[pred > max_depth] = max_depth

	mask = np.logical_and(gt > min_depth, gt < max_depth)
	gt_height, gt_width = gt.shape
	crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
				 0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
	crop_mask = np.zeros(mask.shape)
	crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
	mask = np.logical_and(mask, crop_mask)
	scalar = np.median(gt[mask])/np.median(pred[mask])
	pred[mask] *= scalar

	return pred[mask], gt[mask]



def kitti_compute_metrics(pred, gt):
	"""
	This function computes the metrics value on a pair of (pred, gt).
	Note that the input 'pred' and 'gt' are both nparrays
	Return a list of result float-values which correspond to MAE, MSE, RMSE, and a1, a2, a3
	"""
	# test image pre-processing 
	pred, gt = kitti_metrics_preprocess(pred, gt)

	## compute MSE and RMSE
	mse = ((gt - pred) ** 2).mean()
	rmse = np.sqrt(mse)
	
	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	# compute ap accuracy
	thresh = np.maximum((gt/pred), (pred/gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean((np.abs(gt - pred) / gt))
	sq_rel = np.mean(((gt - pred) ** 2) / gt)
	#print("sq_rel:", sq_rel)

	return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

def nyu_compute_metrics(pred, gt):
	"""
	This function computes the metrics value on a pair of (pred, gt).
	Note that the input 'pred' and 'gt' are both nparrays
	Return a list of result float-values which correspond to MAE, MSE, RMSE, and a1, a2, a3
	"""
	# test image pre-processing 
	pred, gt = nyu_metrics_preprocess(pred, gt)
	#print("pred:", pred)
	#print("gt:", gt)
	#print("++++++++++++++++++++++++++++++++==")

	## compute MSE and RMSE
	mse = ((gt - pred) ** 2).mean()
	rmse = np.sqrt(mse)

	#print("rmse:", rmse)
	
	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	# compute ap accuracy
	thresh = np.maximum((gt/pred), (pred/gt))
	a1 = (thresh < 1.25).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	abs_rel = np.mean((np.abs(gt - pred) / gt))
	sq_rel = np.mean(((gt - pred) ** 2) / gt)
	#print("sq_rel:", sq_rel)
	print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

	return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


def nyu_metrics_preprocess(pred, gt):
	'''
	This function do some preprocessing before metrics calculation
	- check zero value to avoid numerical problems;
	-
	Note that the input 'pred' and 'gt' are both 4D nparrays
	return the corresponding image pair 
	'''
	# squeeze the first and last idx(which is one in test processing)

	pred = np.squeeze(pred)
	gt = np.squeeze(gt)
	#print("gt1:", gt)


	min_depth = 1e-3
	max_depth = 8
	pred[pred < min_depth] = min_depth
	pred[pred > max_depth] = max_depth

	mask = np.logical_and(gt > min_depth, gt < max_depth)

	scalar = np.median(gt[mask])/np.median(pred[mask])
	pred[mask] *= scalar
		# gtiheight, gt_width = gt.shape

	#print("gt2:", gt[mask])
	return pred[mask], gt[mask]


def main():
	global args
	args = parser.parse_args()
		# =========================== DataLoader ===============================
	# syn_dataset "VKITTI"
	# syn_root path
	print("Loading the dataset ...")


	real_loader = loaddata.create_test_dataloader(dataset=args.dataset,
												  root=args.root,
												  data_file=args.test_datafile,
												  batchsize=args.batchSize,
												  nThreads=args.nThreads,
												  loadSize=args.loadSize)
	
	print("Loading data set is complete!")
	print("=======================================================================================")
	print("Building models ...")

	# Define Shared Structure Encoder
	Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
												input_dim=3, dim=64, 
												norm='in', activ='lrelu', 
												pad_type='reflect')
	

	# Define Structure Decoder
	Struct_Decoder = modules.Struct_Decoder()

	# Define Depth-specific Attention (DSA) module

	Attention_Model = DSAModules.drn_d_22(pretrained=True)
	DSAModle = DSAModules.AutoED(Attention_Model)


	# Define DepthNet
	DepthNet = modules.Depth_Net()
	init_weights(DepthNet, init_type='normal')
	
	
	Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()	
	Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
	DSAModle = torch.nn.DataParallel(DSAModle).cuda()
	DepthNet = torch.nn.DataParallel(DepthNet).cuda()	
	
	# Load models
	Shared_Struct_Encoder.load_state_dict(torch.load(args.Shared_Struct_Encoder_path))
	Struct_Decoder.load_state_dict(torch.load(args.Struct_Decoder_path))
	DSAModle.load_state_dict(torch.load(args.DSAModle_path))
	DepthNet.load_state_dict(torch.load(args.DepthNet_path))
	
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
	
	
	if args.dataset == "KITTI":
		Shared_Struct_Encoder.eval()
		Struct_Decoder.eval()
		DSAModle.eval()
		DepthNet.eval()

		result_log = [[] for i in range(7)]

		step = 0

		for i, real_batched in enumerate(real_loader):
			print("step:", step+1)
			image, depth_, depth_interp_ = real_batched['left_img'], real_batched['depth'], real_batched['depth_interp']

			image = torch.autograd.Variable(image).cuda()
			depth_ = torch.autograd.Variable(depth_).cuda()

			# predict
			struct_code = Shared_Struct_Encoder(image)
			structure_map = Struct_Decoder(struct_code)
			
			attention_map = DSAModle(image)

			depth_specific_structure = attention_map * structure_map

			pred_depth = DepthNet(depth_specific_structure)
			pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[depth_.size(1),depth_.size(2)], mode='bilinear',align_corners=True)
	
			pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())
			gt_np = np.squeeze(depth_.cpu().detach().numpy())

			depth_interp_np = np.squeeze(depth_interp_.cpu().detach().numpy())

			pred_depth_np += 1.0
			pred_depth_np /= 2.0
			pred_depth_np *= 80.0

			test_result = kitti_compute_metrics(pred_depth_np, gt_np)   # list1 

			for it, item in enumerate(test_result):
				result_log[it].append(item)

			step = step + 1


		f = open(args.out_dir + "/evalog.txt", 'w')
		f.write('Done testing -- epoch limit reached')
		f.write("after %d iteration \n\n" % (step))
		save_test(f, result_log)
		f.close()


	if args.dataset == "NYUD_V2":
		Shared_Struct_Encoder.eval()
		Struct_Decoder.eval()
		DSAModle.eval()
		DepthNet.eval()

		result_log = [[] for i in range(7)]

		step = 0
		for i, real_batched in enumerate(real_loader):
			print("step:", step+1)
			image, depth_ = real_batched['img'], real_batched['depth']

			image = torch.autograd.Variable(image).cuda()
			depth_ = torch.autograd.Variable(depth_).cuda()

			struct_code = Shared_Struct_Encoder(image)
			structure_map = Struct_Decoder(struct_code)
			attention_map = DSAModle(image)
			depth_specific_structure = attention_map * structure_map
			pred_depth = DepthNet(depth_specific_structure)
			pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[depth_.size(2),depth_.size(3)], mode='bilinear',align_corners=True)
			

			pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())
			gt_np = np.squeeze(depth_.cpu().detach().numpy())

			pred_depth_np += 1.0
			pred_depth_np /= 2.0
			pred_depth_np *= 8.0
			gt_np /= 1000.0


			test_result = nyu_compute_metrics(pred_depth_np, gt_np)   # list1 

			for it, item in enumerate(test_result):

				result_log[it].append(item)

			step = step + 1


		f = open(args.out_dir + "/evalog.txt", 'w')
		f.write('Done testing -- epoch limit reached')
		f.write("after %d iteration \n\n" % (step))
		save_test(f, result_log)
		f.close()


	
if __name__ == '__main__':
	main()


















