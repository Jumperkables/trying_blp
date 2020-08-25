# coding: utf-8
import os, sys

#sys.path.insert(1, os.path.expanduser("~/kable_management/blp_paper/tvqa/mystuff"))
from C3D_model import *
import json
import torchvision
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import collections
#import imageio # read video
import skimage.io as io
from skimage.transform import resize
import h5py
import fnmatch
from PIL import Image
#from visdom_plotter import VisdomLinePlotter

def feature_extractor(args):
	#trainloader = Train_Data_Loader( VIDEO_DIR, resize_w=128, resize_h=171, crop_w = 112, crop_h = 112, nb_frames=16)
	#plotter = VisdomLinePlotter(env_name="c3d Extraction")
	nb_frames = 16
	OUTPUT_DIR = args.OUTPUT_DIR
	EXTRACTED_LAYER = args.EXTRACTED_LAYER
	VIDEO_DIR = args.VIDEO_DIR
	RUN_GPU = args.GPU
	OUTPUT_NAME = args.OUTPUT_NAME
	net = C3D(487)
	print('net', net)
	## Loading pretrained model from sports and finetune the last layer
	net.load_state_dict(torch.load('/home/crhf63/Pytorch_C3D_Feature_Extractor/c3d.pickle'))#/home/jumperkables/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq
	if RUN_GPU : 
		net.cuda(0)
	net.eval()
	print('net', net)
	feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192

	video_list = os.listdir(VIDEO_DIR)
	vlist_2_prefix = {
		'bbt_frames':'',
		'castle_frames':'castle_',
		'met_frames':'met_',
		'grey_frames':'grey_',
		'friends_frames':'friends_',
		'house_frames':'house_'
	}
	print('video_list', video_list)
	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	f = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'w')
		
	def count_files(directory, prefix_list):
		lst = os.listdir(directory)
		#cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
		cnt = len(lst)
		return cnt, lst
	#'/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq'
	clips_to_do = sum([ len(os.listdir(os.path.join('/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq', vid_list))) for vid_list in video_list ])
	import time, datetime

	cumu = 0

	for video_name in video_list: 	
		video_path = os.path.join(VIDEO_DIR, video_name)
		prefix = vlist_2_prefix[video_name]
		print('video_path', video_path)
		cnt, clip_names = count_files(video_path, [])
		clip_paths = [ os.path.join(video_path, clip_name) for clip_name in clip_names ]
		
		index_w = np.random.randint(resize_w - crop_w) ## crop
		index_h = np.random.randint(resize_h - crop_h) ## crop
		#import ipdb; ipdb.set_trace()
		for idx, clip_path in enumerate(clip_paths):
			trace = time.time()	
			clip_name = clip_path.split('/')[-1]
			frames = os.listdir(clip_path)
			frames.sort()
			frame_paths = [ os.path.join(clip_path, frame_path) for frame_path in frames ]
			if len(frame_paths)<16:
				to_copy = frame_paths[-1]
				for x in range(16-len(frame_paths)):
					frame_paths.append(to_copy)
			clip = np.array([ resize(io.imread(frame_path), output_shape=(resize_w, resize_h), preserve_range=True) for frame_path in frame_paths])
			clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
			clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
			clip = Variable(clip).cuda(0) if RUN_GPU else Variable(clip)			
			clip = clip.resize(1, 3, len(frame_paths), crop_w, crop_h)#nb_frames
			to_save = []	# To save everything
			for subfrm in range(len(frame_paths)):
				subclip = to_process(clip, subfrm) 
				_, subclip_output = net(subclip, EXTRACTED_LAYER)
				to_save.append((subclip_output.data).cpu())
			to_save = torch.cat(to_save, dim=0)
			f.create_dataset(prefix+clip_name, data=to_save)
			rate=trace-time.time()
			print(cumu, clips_to_do, str(datetime.timedelta(seconds = -1*int(rate*(clips_to_do-cumu)))))
			#plotter.text_plot("Left", str(cumu)+" "+str(clips_to_do)+" "+str(datetime.timedelta(seconds = -1*int(rate*(clips_to_do-cumu)))))
			cumu += 1

		#features = torch.cat(features, 0)
		#features = features.numpy()
		#print('features', features)
				
		#fgroup = f.create_group(video_name)
		#fgroup.create_dataset('c3d_features', data=features)

		#with open(os.path.join(OUTPUT_DIR, video_name[:-4]), 'wb') as f :
		#	pickle.dump( features, f )
		#print '%s has been processed...'%video_name
	f.close()

def to_process(clip, center):
	# Lengths and frames
	clip_len = clip.shape[2]
	first_frame = clip[:,:,:1]
	last_frame  = clip[:,:,-1:]
	l_pad = max(0, 8-center) #leftpad
	r_pad = max(0, (8+center)-clip_len) #rightpad

	# Prepare chunks to create subclip
	# left
	ret = []
	if l_pad>0:
		first_chunk  = torch.cat([first_frame]*l_pad, dim=2)
		ret.append(first_chunk)
	# middle
	middle_chunk = clip[:,:,max(0,(center-8)):(center+8)]
	ret.append(middle_chunk)
	# right
	if r_pad>0:
		last_chunk	= torch.cat([last_frame]*r_pad, dim=2)
		ret.append(last_chunk)
	return torch.cat(ret, dim=2)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print '******--------- Extract C3D features ------*******'
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='/home/crhf63/kable_management/data/tvqa/motion_features/', help='Output file name')
	parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=6, help='Feature extractor layer')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type = str, default='/home/crhf63/kable_management/data/tvqa/raw_vid/vid_frames/frames_hq/', help='Input Video directory')
	parser.add_argument('-gpu', '--gpu', dest='GPU', action = 'store_true', help='Run GPU?')
	parser.add_argument('--OUTPUT_NAME', default='tvqa_c3d_fc6_features.hdf5', help='The output name of the hdf5 features')
	#python feature_extractor_frm.py -l 6 -i /home/jumperkables/void/raw_vid/vid_frames/frames_hq/ -gpu -id 0 -p /data/miayuan/video_list.txt --OUTPUT_NAME c3d_fc6_features.hdf5

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed parameters:'
	print json.dumps(params, indent = 2)

	OUTPUT_DIR = params['OUTPUT_DIR']
	EXTRACTED_LAYER = params['EXTRACTED_LAYER']
	VIDEO_DIR = params['VIDEO_DIR']
	RUN_GPU = params['GPU']
	OUTPUT_NAME = params['OUTPUT_NAME']
	crop_w = 112
	resize_w = 128
	crop_h = 112
	resize_h = 171
	nb_frames = 16
	feature_extractor(args)


