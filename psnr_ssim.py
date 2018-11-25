import tensorflow as tf
from skimage.measure import compare_ssim
import numpy 
import math
import cv2
import os
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('ground_image_dir', "data/downscaled/" , 'ground_image_dir')
tf.flags.DEFINE_string('output_image_dir', "results/IVC" , 'output_image_dir')


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
	    return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


gt_dir = os.path.join(os.getcwd(), FLAGS.ground_image_dir)
hz_dir = os.path.join(os.getcwd(), FLAGS.output_image_dir)
gt_image_list = os.listdir(gt_dir) 
hz_image_list = os.listdir(hz_dir)

results_arr_psnr = []
results_arr_ssim = []

for gt in gt_image_list:
	for hz in hz_image_list:
		if (gt==hz):
			gt_img = cv2.imread(os.path.join(gt_dir, gt))
			hz_img = cv2.imread(os.path.join(hz_dir, hz))
			(score, diff) = compare_ssim(hz_img, gt_img, full=True, multichannel=True)
			# print("SSIM: {}".format(score))
			d = psnr(gt_img, hz_img)
			results_arr_psnr.append(d)
			results_arr_ssim.append(score)
res_psnr = numpy.array(results_arr_psnr)
res_ssim = numpy.array(results_arr_ssim)
print ("PSNR", numpy.mean(res_psnr))
print ("SSIM", numpy.mean(res_ssim))
