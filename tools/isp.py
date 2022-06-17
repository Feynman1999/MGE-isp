import time
import numpy as np
import rawpy
import sys
import cv2
from scipy.ndimage.filters import convolve
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

pattern = "RGGB"
srgb2xyz = np.mat([     [0.4124564, 0.3575761, 0.1804375],
						[0.2126729, 0.7151522, 0.0721750],
						[0.0193339, 0.1191920, 0.9503041]])
cam2xyz = np.mat([  [0.7188, 0.1641, 0.0781],
					[0.2656, 0.8984, -0.1562],
					[0.0625, -0.4062, 1.1719]])


def masksCFABayer(shape, pattern='RGGB'):
	pattern = pattern.upper()

	channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
	for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
		channels[channel][y::2, x::2] = 1

	return tuple(channels[c].astype(bool) for c in 'RGB')


def masksWB(shape, rwb, gwb, bwb, pattern='RGGB'):
	pattern = pattern.upper()

	channels = np.zeros(shape, dtype = np.float32)
	wb = {'R': rwb, 'G': gwb, 'B': bwb}
	for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
		channels[y::2, x::2] = wb[channel]
	return channels


def wb(o, mask, min, max):
	o = o * mask
	o = np.clip(o, min, max)
	return o


def whiht_balance(x):
	Rm, Gm, Bm = masksCFABayer(x.shape, pattern)

	aver_R  = np.sum(x * Rm) / np.sum(Rm)
	aver_G  = np.sum(x * Gm) / np.sum(Gm)
	aver_B  = np.sum(x * Bm) / np.sum(Bm)
	gray = (aver_B + aver_G + aver_R) / 3

	rwb= gray / aver_R
	gwb = gray / aver_G
	bwb = gray / aver_B
	
	wbm = masksWB(x.shape, rwb, gwb, bwb, pattern)
	x_wb = wb(x, wbm, 0.0001, 1)
	return x_wb


H_G = np.asarray([[0, 1, 0],    [1, 4, 1],    [0, 1, 0]]) / 4

H_RB = np.asarray([[1, 2, 1],    [2, 4, 2],    [1, 2, 1]]) / 4


def demosaic(wb):
	Rm, Gm, Bm = masksCFABayer(wb.shape, pattern)
	R = convolve(wb * Rm, H_RB, mode='mirror')
	G = convolve(wb * Gm, H_G, mode='mirror')
	B = convolve(wb * Bm, H_RB, mode='mirror')
	demosaic_rgb = np.dstack((R,G,B))
	# demosaic_rgb = demosaicing_CFA_Bayer_bilinear(wb, pattern=pattern)
	return demosaic_rgb


def gamma(o, r, min, max):
	o = np.clip(o, min+0.01, max)
	o = np.power(o, 1/r)
	return o


def ccm_gamma(im):
	xyz2srgb = srgb2xyz.I
	cam2srgb =  xyz2srgb * cam2xyz
	cam2srgb_norm= cam2srgb / np.repeat(np.sum(cam2srgb,1), 3).reshape(3,3)
	cmatrix = cam2srgb_norm

	r = cmatrix[0,0] * im[:,:,0] + cmatrix[0,1] * im[:,:,1] + cmatrix[0,2] * im[:,:,2]
	g = cmatrix[1,0] * im[:,:,0] + cmatrix[1,1] * im[:,:,1] + cmatrix[1,2] * im[:,:,2]
	b = cmatrix[2,0] * im[:,:,0] + cmatrix[2,1] * im[:,:,1] + cmatrix[2,2] * im[:,:,2]

	r = np.clip(r, 0.0001, 1)
	g = np.clip(g, 0.0001, 1)
	b = np.clip(b, 0.0001, 1)

	rgb_sRGB = np.dstack((r,g,b))

	gamma_rgb = gamma(rgb_sRGB, 2.5, 0.0001, 1)

	return gamma_rgb


def reshape_raw(bayer):
	bayer = np.expand_dims(bayer,axis=2) 
	bayer_shape = bayer.shape
	H = bayer_shape[0]
	W = bayer_shape[1]
	reshaped = np.concatenate((bayer[0:H:2, 0:W:2, :], 
					   bayer[0:H:2,1:W:2,:],
					   bayer[1:H:2,1:W:2,:],
					   bayer[1:H:2,0:W:2,:]), axis=2)
	return reshaped


def reshape_back_raw(bayer):
	H = bayer.shape[0]
	W = bayer.shape[1]
	newH = int(H*2)
	newW = int(W*2)
	bayer_back = np.zeros((newH, newW))
	bayer_back[0:newH:2,0:newW:2] = bayer[...,0]
	bayer_back[0:newH:2,1:newW:2] = bayer[...,1]
	bayer_back[1:newH:2,1:newW:2] = bayer[...,2]
	bayer_back[1:newH:2,0:newW:2] = bayer[...,3]
	return bayer_back