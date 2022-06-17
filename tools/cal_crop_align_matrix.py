"""
	given a dir of sr-raw dataset, cal tform

	00001.JPG ~ 00007.JPG

	ref: 00001
	resolution: resize according f's proportion
"""
import os
import numpy as np
import cv2
from PIL import Image
import rawpy
from isp import whiht_balance, demosaic, reshape_back_raw, reshape_raw, ccm_gamma


dirpath = "/home/chenyuxiang/repos/00179"
FOCAL_CODE = 37386


def get_bayer(path, black_lv, white_lv):
	try:
		raw = rawpy.imread(path)
	except:
		return None
	bayer = raw.raw_image_visible.astype(np.float32)
	bayer = (bayer - black_lv)/ (white_lv - black_lv) #subtract the black level
	return bayer

def readFocal_pil(image_path):
	if 'ARW' in image_path:
		image_path = image_path.replace('ARW','JPG')
	try:
		img = Image.open(image_path)
	except:
		return None
	exif_data = img._getexif()
	return float(exif_data[FOCAL_CODE]._val)


def crop_fov(image, ratio):
	h, w , _ = image.shape
	new_width = int(np.floor(w * ratio + 0.5))
	new_height = int(np.floor(h * ratio + 0.5))
	left = np.ceil((w - new_width)/2.)
	top = np.ceil((h - new_height)/2.)
	# print("Cropping boundary: ", top, bottom, left, right)
	cropped = image[int(top):int(top) + new_height, int(left):int(left) + new_width, ...]
	return cropped


def align_ecc(images_gray_set, ref_ind, thre=0.05, mode = 'Homo'):
	img_num = len(images_gray_set)
	ref_gray_image = images_gray_set[ref_ind]
	r, c = images_gray_set[0].shape[0:2]

	if mode == 'Homo':
		warp_mode = cv2.MOTION_HOMOGRAPHY # 3*3
	else:
		warp_mode = cv2.MOTION_AFFINE  # 2*3
		
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		identity_transform = np.eye(3, 3, dtype=np.float32)
		tform_set = np.zeros((img_num, 3, 3), dtype=np.float32)
		tform_inv_set = np.zeros_like(tform_set)
	else :
		identity_transform = np.eye(2, 3, dtype=np.float32)
		tform_set = np.zeros((img_num, 2, 3), dtype=np.float32)
		tform_inv_set = np.zeros_like(tform_set)
	
	number_of_iterations = 500
	termination_eps = 1e-6
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

	# Run the ECC algorithm. The results are stored in warp_matrix.
	motion_thre = thre * min(r, c)

	for i in range(0, img_num, 1): # forward
		if warp_mode == cv2.MOTION_HOMOGRAPHY :
			warp_matrix = np.eye(3, 3, dtype=np.float32)
		else:
			warp_matrix = np.eye(2, 3, dtype=np.float32)
		_, warp_matrix = cv2.findTransformECC(ref_gray_image, images_gray_set[i], warp_matrix, warp_mode, criteria)
		
		tform_set[i] = warp_matrix
		if warp_mode == cv2.MOTION_HOMOGRAPHY :
			_, tform_inv_set[i] = cv2.invert(warp_matrix)
		else:
			tform_inv_set[i] = cv2.invertAffineTransform(warp_matrix)

		motion_val = abs(warp_matrix - identity_transform).sum()
		if motion_val < motion_thre:
			pass
		else:
			raise RuntimeError("")
	return tform_set, tform_inv_set


def back_to_rgb(x):
	x = reshape_back_raw(x)
	x = whiht_balance(x)
	x = demosaic(x)
	x = ccm_gamma(x)
	x = x[:, :, ::-1].astype(np.float32)
	return x

if __name__ == "__main__":
	rsz = 2
	raw_imgs = []
	focals = []
	resized_imgs = []
	resized_gray_imgs = []

	desti = "/home/chenyuxiang/repos/00179/myisp_no_rgb_align"
	if not os.path.exists(desti):
		os.mkdir(desti)

	for i in range(1, 8):
		imgpath = os.path.join(dirpath, "{}.JPG".format(str(i).zfill(5)))
		focal = readFocal_pil(imgpath)
		focals.append(focal)

		# get raw data
		path_raw = os.path.join(dirpath, "{}.ARW".format(str(i).zfill(5)))
		input_raw = get_bayer(path_raw, black_lv = 512, white_lv = 16383)
		H,W = input_raw.shape
		reshaped_raw = reshape_raw(input_raw) # h/2, w/2, 4
		croped_rawimg = crop_fov(reshaped_raw, ratio=focal / focals[0])

		raw_imgs.append(croped_rawimg)

		croped_rgbimg = back_to_rgb(croped_rawimg)
		
		dsize = (W // 2, H // 2)
		dsize_align = (W// (2**rsz), H // (2**rsz))

		# # resize croped img
		resized_croped_img = cv2.resize(croped_rawimg, dsize=dsize, interpolation=cv2.INTER_CUBIC)
		resized_croped_img_align = cv2.resize(croped_rgbimg, dsize=dsize_align, interpolation=cv2.INTER_CUBIC)

		resized_imgs.append(resized_croped_img)
		resized_gray_imgs.append(cv2.cvtColor(resized_croped_img_align, cv2.COLOR_BGR2GRAY))
		cv2.imwrite("{}/before_{}.png".format(desti, i), (resized_croped_img_align*255).astype(np.uint8))

	# t  ref -> other
	# t_inv  other -> ref
	t, t_inv = align_ecc(resized_gray_imgs, ref_ind = 0, thre=0.3)
	
	# # align imgs to ref
	for i in range(2, 8):
		# align in original 
		img = cv2.warpPerspective(resized_gray_imgs[i-1], t_inv[i-1], dsize=dsize_align)
		# img = cv2.warpAffine(resized_gray_imgs[i-1], t_inv[i-1], dsize=dsize_align) 
		cv2.imwrite("{}/{}.png".format(desti, i), (img*255).astype(np.uint8))


	# scale back
	t[:, 0, 2] *= (2**rsz) / 2
	t[:, 1, 2] *= (2**rsz) / 2
	t_inv[:, 0, 2] *= (2**rsz) / 2
	t_inv[:, 1, 2] *= (2**rsz) / 2
	

	# save 1 align to 5
	lr_index = 3
	# img = cv2.warpAffine(resized_imgs[0], t[lr_index], dsize=dsize) 
	img = cv2.warpPerspective(resized_imgs[0], t[lr_index], dsize=dsize)
	LR = raw_imgs[lr_index] 
	h,w,_ = LR.shape
	# resize to 4x of 6
	HR = cv2.resize(img, dsize=(4*w, 4*h), interpolation=cv2.INTER_CUBIC)
	print(LR.shape, HR.shape)
	LR_rgb = back_to_rgb(LR)
	HR_rgb = back_to_rgb(HR)
	cv2.imwrite("{}/LR.png".format(desti), (LR_rgb*255).astype(np.uint8))
	cv2.imwrite("{}/HR.png".format(desti), (HR_rgb*255).astype(np.uint8))
	

