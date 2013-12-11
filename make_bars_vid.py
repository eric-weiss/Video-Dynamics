import numpy as np
import cPickle as cp

from matplotlib import pyplot as pp

from makegif import writeGif

nbars=6
nx=64
wmin=1
wmax=5
smin=2
smax=8

nt=50

pics=[]

x=np.arange(-nx/2.0,nx/2.0)
X,Y=np.meshgrid(x,x)

p0=-nx
pos=p0*np.ones(nbars)
vecs=np.random.randn(nbars,2)
vecs=vecs/np.reshape(np.sqrt(np.sum(vecs**2,axis=1)),(nbars,1))
widths=np.random.rand(nbars)*(wmax-wmin)+wmin
speeds=np.random.rand(nbars)*(smax-smin)+smin

print pos
print vecs
print widths
print speeds

for i in range(nt):
	
	#draw the bars
	pic=np.zeros((nx,nx))
	for j in range(nbars):
		
		cx=vecs[j,0]*pos[j]; cy=vecs[j,1]*pos[j]
		fx=cx+vecs[j,0]*widths[j]; fy=cy+vecs[j,1]*widths[j]
		bx=cx-vecs[j,0]*widths[j]; by=cy-vecs[j,1]*widths[j]
		
		fdx=X-fx; fdy=Y-fy; bdx=X-bx; bdy=Y-by
		
		fdots=fdx*vecs[j,0]+fdy*vecs[j,1]
		bdots=bdx*vecs[j,0]+bdy*vecs[j,1]
		
		barpic=np.clip(np.exp(-(fdots*bdots)),0,1)
		
		pic=pic+barpic
	
	pic=np.clip(pic, 0,1)
	
	pics.append(pic)
	pos=pos+0.5*speeds

writeGif('test.gif',pics)
