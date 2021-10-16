import numpy as np
import cv2
import argparse
from baselines.common.trex_utils import mask_score


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--data_dir', default='../../atari-head', help='data directory')
parser.add_argument('--env_name', default='centipede')
parser.add_argument('--trial_name', default='501_RZ_3566169_Jul-19-14-57-05')
parser.add_argument('--out_path', default='./gaze_videos')
parser.add_argument('--mask', action='store_true', default=False)

args = parser.parse_args()
env_name = args.env_name
data_dir = args.data_dir
trial_name = args.trial_name
out_path = args.out_path

#trial = '../data/atari-head/hero/195_RZ_205678_Jun-28-12-09-00'
#trial = '../data/atari-head/mspacman/209_RZ_6964528_Jan-08-10-23-46'
trial = data_dir + '/' + env_name + '/' + trial_name

img_folder = trial + '/'
#img_folder = '../data/atari-head/hero/195_RZ_205678_Jun-28-12-09-00/'
f = open(trial+".txt")
fps = 20
# format="XVID" #avi
format = "mp4v"
outvid= out_path+'/'+env_name+'_'+trial_name+'.mp4'
print(outvid)

fourcc = cv2.VideoWriter_fourcc(*format)
vid = None
size = None
is_color = True


action_name = { 
'0': 'PLAYER_A_NOOP',

'1': 'PLAYER_A_FIRE',          
'2': 'PLAYER_A_UP',             
'3': 'PLAYER_A_RIGHT',          
'4': 'PLAYER_A_LEFT',           
'5': 'PLAYER_A_DOWN',          

'6': 'PLAYER_A_UPRIGHT',        
'7': 'PLAYER_A_UPLEFT',         
'8': 'PLAYER_A_DOWNRIGHT',     
'9': 'PLAYER_A_DOWNLEFT',       

'10': 'PLAYER_A_UPFIRE',        
'11': 'PLAYER_A_RIGHTFIRE',     
'12': 'PLAYER_A_LEFTFIRE',      
'13': 'PLAYER_A_DOWNFIRE',     

'14': 'PLAYER_A_UPRIGHTFIRE',   
'15': 'PLAYER_A_UPLEFTFIRE',    
'16': 'PLAYER_A_DOWNRIGHTFIRE', 
'17': 'PLAYER_A_DOWNLEFTFIRE',
'null': 'NULL'
}

font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,10)
bottomLeftCornerOfText = (10,180)
bottomRightCornerOfText = (130,180)
fontScale              = 0.25
fontColor              = (255,255,255)
lineType               = 1

line = f.readline()
# print(line)
i = 0
blink = False
#speed = [0,0]
#previous_gaze = [0,0]
counter = 0
colors = [(0,255,0),(0,0,255)]

k=0
gaze_ignore = []
for line in f:
	k+=1
	gaze_ignore.append(0)
	contents = line.split(',')
	# if (i==0):
			# print(contents[0])
	# i = 1
	img_name = contents[0]
	episode = contents[1]
	score = contents[2]
	duration = contents[3]
	unclipped_reward = contents[4]
	action = contents[5]
	gaze = contents[6:]

	img = cv2.imread(img_folder+img_name+'.png')

	for j in range(0,len(gaze),2):
		if('null' not in gaze[j]):
			x = float(gaze[j])
			y = float(gaze[j+1])
		if(y>200):
			blink = True
	if blink:
		#if(len(gaze_ignore))>10:    
		#	gaze_ignore[k-10:k] = np.ones(60)	
		blink=False 

f.close()
f = open(trial+".txt")
line = f.readline()
blink = False
k = 0
for line in f:	
	contents = line.split(',')
	# if (i==0):
		# print(contents[0])
	# i = 1
	img_name = contents[0]
	episode = contents[1]
	score = contents[2]
	duration = contents[3]
	unclipped_reward = contents[4]
	action = contents[5]
	gaze = contents[6:]

	img = cv2.imread(img_folder+img_name+'.png')
	# print('img_read')

	if vid is None:
		if size is None:
			size = img.shape[1], img.shape[0]
		vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
		# if size[0] != img.shape[1] and size[1] != img.shape[0]:
		# 	img = resize(img, size)
		# mask image of size 1,84,84,3
	if args.mask:
		mask_img = cv2.resize(img,(84,84))
		mask_img = np.expand_dims(mask_img, axis=0)
		mask_img = mask_score(mask_img,env_name)
		mask_img = np.squeeze(mask_img,axis=0)
		img = cv2.resize(mask_img,size)

	# overlay gaze coordinates on img
	for j in range(0,len(gaze),2):
		if('null' not in gaze[j]):
			x = float(gaze[j])
			y = float(gaze[j+1])
		if(y>200):
			blink = True 
			counter = 0
			text_color = colors[(i+1)%2]
		if blink:
			counter+=1
			#cv2.putText(img,'BLINKING',
			#			bottomLeftCornerOfText, font, fontScale, text_color, lineType)	
		if counter==300 and blink==True:
			blink=False

		gaze_coord_text = '('+str(int(x))+','+str(int(y))+')'
		#cv2.putText(img, gaze_coord_text, bottomRightCornerOfText, 
		#	font, 0.2, (0,255,0), lineType)
		
		if gaze_ignore[k]==0:
			cv2.circle(img, (int(x),int(y)), 5, (0,255,0), thickness=1, lineType=8, shift=0)

		# TODO: show action and return on the video
		#print(action_name[action])
		# cv2.putText(img,action_name[action], 
		# 	topLeftCornerOfText, 
		# 	font, 
		# 	fontScale,
		# 	fontColor,
		# 	lineType)
	#i+=1
	k+=1

	vid.write(img)
vid.release()
# return vid



# TODO: repeat frames if the duration of gaze on a frame is >1/20 seconds. Repeat it n/20 times.
# TODO: alternate color on text for consecutive blink detections
