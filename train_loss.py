# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os
import torch
import torch.nn as nn
import torch.nn.functional as F





def depth_loss(output, depth_gt):

	losses=[]

	depth_loss = torch.nn.L1Loss()

	for depth_index in range(len(output)):

		loss = depth_loss(output[depth_index], depth_gt[depth_index])

		losses.append(loss)


	total_loss = sum(losses)
	
	return total_loss


def gradient_x(img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

def gradient_y(img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy




def struct_weighted_loss(struct_map, depth, train_iteration, hyper_w):

	depth_grad_dx = gradient_x(depth)
	
	depth_grad_dy = gradient_y(depth)
	
	depth_grad = torch.abs(depth_grad_dx) + torch.abs(depth_grad_dy)
	
	weight = torch.exp(-torch.mean(depth_grad, 1, keepdim=True) * hyper_w)
	
	weighted_struct = struct_map * weight

	return torch.mean(torch.abs(weighted_struct))