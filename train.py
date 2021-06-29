# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os 
import time
import torch
import argparse
import loaddata
import DSAModules
import itertools
import train_loss
import numpy as np
from utils import *
import torch.nn as nn
import torch.nn.parallel
from models import modules
from itertools import chain
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

## =========================== Parameters =================
parser = argparse.ArgumentParser(description="Domian transfer on depth estimation.")
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')		
parser.add_argument('--syn_dataset', type=str, default='VKITTI', help='synthetic domain')                       # *******
parser.add_argument('--syn_root',  type=str, default='', help='path to source dataset.')                        # *******
parser.add_argument('--syn_train_datafile', type=str, default='', help='stores data list, in syn_root')         # *******
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')                                # *******
parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')                       # *******
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')           # *******
parser.add_argument('--no_rotation', action='store_true', help='if specified, do not rotate the images for data augmentation')     # *******
parser.add_argument('--no_augment', action='store_true', help='if specified, do not use data augmentation, e.g., randomly shifting gamma')  # *******
parser.add_argument('--loadSize', nargs='+', type=int, default=286, help='scale images to this size')           # *******
parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints/", help='the path of saving models')
parser.add_argument('--log_dir', type=str, default="./log", help="the path of log")
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate of network')                # *******
parser.add_argument('--Shared_Struct_Encoder_path', type=str, default='', help='the path of shared geo encoder')
parser.add_argument('--Struct_Decoder_path', type=str, default='', help="the path of Struct_Decoder")
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lambda_w', type=float, default=1.0, help='the weight parameters of structure map.')
parser.add_argument('--hyper_w', type=float, default=0.001, help='the weight parameters.')
parser.add_argument('--train_stage', type=str, default='TrainStructDecoder', help='train stage(TrainStructDecoder/TrainDSAandDPModule).')


def main():
	global args
	args = parser.parse_args()
	# make log_dir and checkpoint_dir
	makedir(args.log_dir)
	makedir(args.checkpoint_dir)
	# =========================== DataLoader ===============================
	# syn_dataset "VKITTI"
	# syn_root path
	print("Loading the dataset ...")
	synthetic_loader = loaddata.creat_train_dataloader(dataset=args.syn_dataset,
													   root=args.syn_root,
													   data_file=args.syn_train_datafile,
													   batchsize=args.batchSize,
													   nThreads=args.nThreads,
													   no_flip=args.no_flip,
													   no_rotation=args.no_rotation,
													   no_augment=args.no_augment,
													   loadSize=args.loadSize)

	print("Loading datasets is complete!")
	print("=======================================================================================")
	print("Building models ...")

	### ================================= STE Module ==================================================

	# Define Shared Structure Encoder
	Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
											input_dim=3, dim=64, 
											norm='in', activ='lrelu', 
											pad_type='reflect')
	# Define Structure Decoder
	Struct_Decoder = modules.Struct_Decoder()

	### ================================ DSA Module ===================================================

	# Define Depth-specific Attention (DSA) module
	Attention_Model = DSAModules.drn_d_22(pretrained=True)
	DSAModle = DSAModules.AutoED(Attention_Model)

	# Define DepthNet
	DepthNet = modules.Depth_Net()
	init_weights(DepthNet, init_type='normal')


	cudnn.enabled = True
	cudnn.benchmark = True

	if args.train_stage == 'TrainStructDecoder':
		## Load pretrained shared_geo_encoder
		Shared_Struct_Encoder.load_state_dict(torch.load(args.Shared_Struct_Encoder_path))

		# =============================== Multi-GPU ======================
		print("GPU num:", torch.cuda.device_count())
		if torch.cuda.device_count() == 8:
			Shared_Struct_Encoder = torch.nn.DataParallel(Shared_Struct_Encoder, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			Struct_Decoder = torch.nn.DataParallel(Struct_Decoder, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			DepthNet = torch.nn.DataParallel(DepthNet, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()

		elif torch.cuda.device_count() == 4:
			Shared_Struct_Encoder = torch.nn.DataParallel(Shared_Struct_Encoder, device_ids=[0, 1, 2, 3]).cuda()
			Struct_Decoder = torch.nn.DataParallel(Struct_Decoder, device_ids=[0, 1, 2, 3]).cuda()
			DepthNet = torch.nn.DataParallel(DepthNet, device_ids=[0, 1, 2, 3]).cuda()

		else:
			Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()
			Struct_Decoder = Struct_Decoder.cuda()
			DepthNet = DepthNet.cuda()
		
	elif args.train_stage == 'TrainDSAandDPModule':
		## Load pretrained shared_geo_encoder
		Shared_Struct_Encoder.load_state_dict(torch.load(args.Shared_Struct_Encoder_path))

		# =============================== Multi-GPU ======================
		print("GPU num:", torch.cuda.device_count())
		if torch.cuda.device_count() == 8:
			Shared_Struct_Encoder = torch.nn.DataParallel(Shared_Struct_Encoder, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			Struct_Decoder = torch.nn.DataParallel(Struct_Decoder, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			DSAModle = torch.nn.DataParallel(DSAModle, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			DepthNet = torch.nn.DataParallel(DepthNet, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()

		elif torch.cuda.device_count() == 4:
			Shared_Struct_Encoder = torch.nn.DataParallel(Shared_Struct_Encoder, device_ids=[0, 1, 2, 3]).cuda()
			Struct_Decoder = torch.nn.DataParallel(Struct_Decoder, device_ids=[0, 1, 2, 3]).cuda()
			DSAModle = torch.nn.DataParallel(DSAModle, device_ids=[0, 1, 2, 3]).cuda()
			DepthNet = torch.nn.DataParallel(DepthNet, device_ids=[0, 1, 2, 3]).cuda()

		else:
			Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()
			Struct_Decoder = Struct_Decoder.cuda()
			DSAModle = DSAModle.cuda()
			DepthNet = DepthNet.cuda()

		## Load Struct_Decoder
		Struct_Decoder.load_state_dict(torch.load(args.Struct_Decoder_path))

	if args.train_stage == 'TrainStructDecoder':
		# =============================== Optim ============================================
		optimizer = torch.optim.Adam(itertools.chain(Struct_Decoder.parameters(), DepthNet.parameters()), lr=args.lr, betas=(0.9, 0.999))
	elif args.train_stage == 'TrainDSAandDPModule':
		# =============================== Optim ============================================
		optimizer = torch.optim.Adam(itertools.chain(DSAModle.parameters(), DepthNet.parameters()), lr=args.lr, betas=(0.9, 0.999))

	# Set logger
	writer = SummaryWriter(log_dir=args.log_dir)

	# Set scheduler
	scheduler = get_scheduler(optimizer, args)
	
	lr = args.lr

	#   train process
	for epoch in range(args.start_epoch, args.epochs):

		batch_time = AverageMeter()
		losses = AverageMeter()

		if args.train_stage == 'TrainStructDecoder':
			Shared_Struct_Encoder.eval()
			Struct_Decoder.train()
			DepthNet.train()

		elif args.train_stage == 'TrainDSAandDPModule':
			Shared_Struct_Encoder.eval()
			Struct_Decoder.eval()
			DSAModle.train()
			DepthNet.train()

		end = time.time()

		for i, sample_batched in enumerate(synthetic_loader):
			image, depth = sample_batched['src']['img'], sample_batched['src']['depth']
			train_iteration = epoch * len(synthetic_loader) + i

			image = torch.autograd.Variable(image).cuda()       # image
			depth = torch.autograd.Variable(depth).cuda()       # depth
			if args.train_stage == 'TrainStructDecoder':
 				# Fix Shared Structure Encoder
				struct_code = Shared_Struct_Encoder(image).detach()
			elif args.train_stage == 'TrainDSAandDPModule':
				# Fix Shared Structure Encoder
				struct_code = Shared_Struct_Encoder(image).detach()
				structure_map = Struct_Decoder(struct_code).detach()
				
			optimizer.zero_grad()

			if args.train_stage == 'TrainStructDecoder':
				structure_map = Struct_Decoder(struct_code)
				pred_depth = DepthNet(structure_map)

			elif args.train_stage == 'TrainDSAandDPModule':
				attention_map = DSAModle(image)
				depth_specific_structure = attention_map * structure_map
				pred_depth = DepthNet(depth_specific_structure)


			gt_depth = adjust_gt(depth, pred_depth)


			depth_loss = train_loss.depth_loss(pred_depth, gt_depth)
			if args.train_stage == 'TrainStructDecoder':
				struct_weighted_loss = train_loss.struct_weighted_loss(structure_map, depth, train_iteration, args.hyper_w)
				total_loss = depth_loss + args.lambda_w * struct_weighted_loss
			elif args.train_stage == 'TrainDSAandDPModule':
				total_loss = depth_loss

			losses.update(total_loss.item(), image.size(0))
			total_loss.backward()
			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()

			batchSize = depth.size(0)
			if train_iteration % 30 == 0:
				writer.add_scalar('train/total_loss', total_loss, train_iteration)
				writer.add_scalar('train/batches_loss_avg', losses.avg, train_iteration)
				writer.add_scalar('train/depth_loss', depth_loss, train_iteration)
				if args.train_stage == 'TrainStructDecoder':
					writer.add_scalar('train/struct_weighted_loss', struct_weighted_loss, train_iteration)

				writer.add_image('train/image', vutils.make_grid(image*0.5+0.5), train_iteration)
				writer.add_image('train/pred_depth', vutils.make_grid(colormap(pred_depth[-1])), train_iteration)
				writer.add_image('train/depth_gt', vutils.make_grid(colormap(depth)), train_iteration)
				writer.add_image('train/structure_map', vutils.make_grid(colormap(structure_map, 'viridis')), train_iteration)
				if args.train_stage == 'TrainDSAandDPModule':
					writer.add_image('train/attention_map', vutils.make_grid(colormap(attention_map, 'viridis')), train_iteration)
					writer.add_image('train/depth_specific_structure', vutils.make_grid(colormap(depth_specific_structure, 'viridis')), train_iteration)


			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'
				  .format(epoch, i, len(synthetic_loader), batch_time=batch_time, loss=losses))

		lr = update_learning_rate(optimizer, scheduler)   
		if (epoch+1) % 1 == 0:
			if args.train_stage == 'TrainStructDecoder':
				torch.save(Struct_Decoder.state_dict(), args.checkpoint_dir + 'struct_decoder_'+str(epoch+1) + ".pth")
				torch.save(DepthNet.state_dict(), args.checkpoint_dir + 'depth_net_'+str(epoch+1) + ".pth")
			if args.train_stage == 'TrainDSAandDPModule':
				torch.save(DSAModle.state_dict(), args.checkpoint_dir + 'dsa_modle_'+str(epoch+1) + ".pth")
				torch.save(DepthNet.state_dict(), args.checkpoint_dir + 'depth_net_'+str(epoch+1) + ".pth")




if __name__ == '__main__':
	main()
