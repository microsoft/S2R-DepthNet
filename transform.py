# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import math
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class RandomHorizontalFlip(object):
	def __init__(self, prob=None):
		self.prob = prob
	def __call__(self, img):
		if (self.prob is None and random.random()<0.5) or self.prob<0.5:
			return img.transpose(Image.FLIP_LEFT_RIGHT)
		return img

class RandomVerticalFlip(object):
	def __init__(self, img):
		if (self.prob is None and random.random()<0.5) or self.prob < 0.5:
			return img.transpose(Image.FLIP_TOP_BOTTOM)
		return img


class RandomImgAugment(object):

	def __init__(self, no_flip, no_rotation, no_augment, size=None, dataset=None):
		self.flip = not no_flip                          ## default all run
		self.augment = not no_augment
		self.rotation = not no_rotation 
		self.size = size
		self.dataset = dataset

	def __call__(self, inputs):
		img1 = inputs[0]                            # Image
		img2 = inputs[1]                            # None
		depth = inputs[2]                           # Depth
		phase = inputs[3]                           # train/test
		fb = inputs[4]                              # focallength*baseline

		h = img1.height                             # height
		w = img1.width								# width
		w0 = w                                      # w0
		
		if self.size == [-1]:                       
			divisor = 32.0                                      # divisor                  
			h = int(math.ceil(h/divisor) * divisor)
			w = int(math.ceil(w/divisor) * divisor)
			self.size = (h, w)
		

		## resize to 256 1024
		scale_transform = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
		img1 = scale_transform(img1)              ## RGB image
		if img2 is not None:                      
			img2 = scale_transform(img2)
		if fb is not None:                              ## fb is None
			scale = float(self.size[1]) / float(w0)     ## 
			fb = fb * scale
		if phase == 'test':                             ## phase train
			return img1, img2, depth, fb
		if depth is not None:
			scale_transform_d = transforms.Compose([transforms.Resize(self.size, Image.BICUBIC)])
			depth = scale_transform_d(depth)
		if not self.size == 0:
			if depth is not None:
				if self.dataset.upper() == 'KITTI' or self.dataset.upper() == 'VKITTI':
					#print("Using outdoor scene transform.")
					arr_depth = np.array(depth, dtype=np.float32)
					arr_depth[arr_depth>8000.0]=8000.0
					arr_depth /= 8000.0   # cm -> m 
					arr_depth[arr_depth<0.0] = 0.0
				if self.dataset.upper() == 'NYUD_V2' or self.dataset.upper() == 'SUNCG':
					#print("Using indoor scene transform.")
					arr_depth = np.array(depth, dtype=np.float32)
					arr_depth /= 255.0       ## normalize to (0, 1)
					arr_depth[arr_depth<0.0] = 0.0

				depth = Image.fromarray(arr_depth, 'F')
		## random horizontal flip
		if self.flip and not (img2 is not None and depth is not None):
			flip_prob = random.random()
			flip_transform = transforms.Compose([RandomHorizontalFlip(flip_prob)])
			if img2 is None:
				img1 = flip_transform(img1)
			else:
				if flip_prob < 0.5:
					img1_ = img1
					img2_ = img2
					img1 = flip_transform(img2_)
					img2 = flip_transform(img1_)

			if depth is not None:
				depth = flip_transform(depth)


		### rotation
		if self.rotation and not (img2 is not None and depth is not None):
			if random.random() < 0.5:
				degree = random.randrange(-500, 500)/100
				img1 = F.rotate(img1, degree, Image.BICUBIC)
				if depth is not None:
					depth = F.rotate(depth, degree, Image.BICUBIC)
				if img2 is not None:
					img2 = F.rotate(img2, degree, Image.BICUBIC)
		# convert depth to range [-1, 1]
		if depth is not None:
			depth = np.array(depth, dtype=np.float32)
			depth = depth * 2.0
			depth -= 1.0

		if self.augment:
			if random.random() < 0.5:

				brightness = random.uniform(0.8, 1.0)
				contrast = random.uniform(0.8, 1.0)
				saturation = random.uniform(0.8, 1.0)

				img1 = F.adjust_brightness(img1, brightness)
				img1 = F.adjust_contrast(img1, contrast)
				img1 = F.adjust_saturation(img1, saturation)

				if img2 is not None:
					img2 = F.adjust_brightness(img2, brightness)
					img2 = F.adjust_contrast(img2, contrast)
					img2 = F.adjust_saturation(img2, saturation)

		return img1, img2, depth, fb

























