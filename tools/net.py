import megengine as mge
import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d
import megengine.functional as F
import math

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow, border_mode):
	"""
		CONSTANT(0)    REPLICATE
	"""
	_, _, H, W = tenFlow.shape
	if str(tenFlow.shape) not in backwarp_tenGrid.keys():
		x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
		x_list = x_list.repeat(H, axis=2)
		y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
		y_list = y_list.repeat(W, axis=3)
		xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
		backwarp_tenGrid[str(tenFlow.shape)] = megengine.tensor(xy_list.astype(np.float32))
	return F.nn.remap(inp = tenInput, map_xy=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).transpose(0, 2, 3, 1), border_mode=border_mode)

class Basic(M.Module):
	def __init__(self, intLevel):
		super(Basic, self).__init__()
		self.netBasic = M.Sequential(
			Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), # 8=3+3+2
			M.ReLU(),
			Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
			M.ReLU(),
			Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
			M.ReLU(),
			Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
			M.ReLU(),
			Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
		)

	def forward(self, tenInput):
		return self.netBasic(tenInput)

class Spynet(M.Module):
	def __init__(self, num_layers, pretrain_ckpt_path = None):
		super(Spynet, self).__init__()
		assert num_layers in (6, 5, 4)
		self.num_layers = num_layers
		self.threshold = 2
		self.pretrain_ckpt_path = pretrain_ckpt_path

		basic_list = [ Basic(intLevel) for intLevel in range(num_layers) ]
		self.border_mode = "REPLICATE"
		self.netBasic = M.Sequential(*basic_list)

		self.init_weights()

	def preprocess(self, tenInput):
		tenRed = (tenInput[:, 0:1, :, :]*0.5 + 0.5 - 0.485) / 0.229
		tenGreen = (tenInput[:, 1:2, :, :]*0.5 + 0.5 - 0.456) / 0.224
		tenBlue = (tenInput[:, 2:3, :, :]*0.5 + 0.5 - 0.406 ) / 0.225
		return F.concat([tenRed, tenGreen, tenBlue], axis=1) # [B,3,H,W]

	def forward(self, tenFirst, tenSecond):
		_,_,H,W = tenFirst.shape
		aim_H = 640 * 2
		aim_W = 960 * 2
		tenFirst = F.nn.interpolate(tenFirst, size=[aim_H, aim_W], align_corners=False)
		tenSecond = F.nn.interpolate(tenSecond, size=[aim_H, aim_W], align_corners=False)

		tenFirst = [self.preprocess(tenFirst)]
		tenSecond = [self.preprocess(tenSecond)]

		for intLevel in range(self.num_layers - 1):
			if tenFirst[0].shape[2] >= self.threshold or tenFirst[0].shape[3] >= self.threshold:
				tenFirst.insert(0, F.avg_pool2d(inp=tenFirst[0], kernel_size=2, stride=2))
				tenSecond.insert(0, F.avg_pool2d(inp=tenSecond[0], kernel_size=2, stride=2))
		
		tenFlow = F.zeros([tenFirst[0].shape[0], 2, tenFirst[0].shape[2], tenFirst[0].shape[3]])
		tenUpsampled = tenFlow
		tenFlow = self.netBasic[0]( F.concat([tenFirst[0], backwarp(tenInput=tenSecond[0], tenFlow=tenUpsampled, border_mode=self.border_mode), tenUpsampled], axis=1) ) + tenUpsampled
		for intLevel in range(1, len(tenFirst)):
			tenUpsampled = F.nn.interpolate(inp=tenFlow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
			tenFlow = self.netBasic[intLevel]( F.concat([tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled, border_mode=self.border_mode), tenUpsampled], axis=1) ) + tenUpsampled
		
		tenFlow = F.nn.interpolate(inp=tenFlow, size=[H, W], align_corners=False)
		tenFlow[:, 0, :, :] *= float(W) / float(aim_W)
		tenFlow[:, 1, :, :] *= float(H) / float(aim_H)

		return tenFlow

	def init_weights(self, strict=True):
		if self.pretrain_ckpt_path is not None:
			print("loading pretrained model for Spynet 🤡🤡🤡🤡🤡🤡...")
			state_dict = megengine.load(self.pretrain_ckpt_path)
			# print(state_dict.keys())
			new_dict = {}
			for item in state_dict.keys():
				if 'flow' in item:
					new_key = ".".join(item.split(".")[1:])
					new_dict[new_key] = state_dict[item]
			self.load_state_dict(new_dict, strict=strict)
		else:
			raise RuntimeError("")