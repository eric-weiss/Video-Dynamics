import numpy as np
import cPickle as cp
import Image
from utils import tile_raster_images
from matplotlib import pyplot as pp
import matplotlib.cm as cm


#This script whitens entire video frames and saves 
#the whitened video as blocks in a single cpickle file

out_fn=raw_input('Output filename: ')

X,Y=np.meshgrid(np.arange(-320,320,1),np.arange(-240,240,1))

filt=np.sqrt((X/320.0)**2+(Y/240.0)**2)*np.exp(-0.5*(np.sqrt((X/320.0)**2+(Y/240.0)**2)/0.8)**4)


pp.matshow(filt,cmap=cm.gray)
pp.show()

datadir='/home/float/Desktop/Bruno_rotation_project/frames/'

blocksize=128
block_counter=0
frame_counter=1

f=open('/home/float/Desktop/Bruno_rotation_project/data/whitened/'+out_fn+'.cpl','wb')
block=[]
while True:
	try:
		im=Image.open(datadir+'frame'+(str)(frame_counter)+'.png').convert('L')
		im=np.asarray(im)
		
		tr=np.fft.fftshift(np.fft.fft2(im))
		tr=tr*filt
		wht=np.fft.ifft2(np.fft.fftshift(tr))
		block.append(np.real(wht))
		
		if frame_counter==1:
			pp.matshow(np.real(wht),cmap=cm.gray)
			pp.show()
		
		if frame_counter%blocksize==0:
			block=np.asarray(block,dtype='float32')
			cp.dump(block,f,2)
			block=[]
			print frame_counter
		
		frame_counter+=1
	
	except IOError:
		break


if len(block)!=0:
	cp.dump(block,f,2)

f.close()
