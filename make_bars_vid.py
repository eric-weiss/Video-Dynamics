import numpy as np
import cPickle as cp

from matplotlib import pyplot as pp

from makegif import writeGif

nbars=3
nx=8
wmin=0.5
wmax=1
smin=1
smax=3

nt=100000

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
out=[]

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
	out.append(pic.flatten())
	
	for j in range(nbars):
		if pos[j]>nx:
			pos[j]=-nx
			vec=np.random.randn(1,2)
			vec=vec/np.reshape(np.sqrt(np.sum(vec**2,axis=1)),(1,1))
			width=np.random.rand()*(wmax-wmin)+wmin
			speed=np.random.rand()*(smax-smin)+smin
			vecs[j]=vec
			widths[j]=width
			speeds[j]=speed

out=np.asarray(out,dtype='float32')
f=open('data/bars.cpl','wb')
cp.dump(out,f,2)
f.close()

writeGif('test.gif',pics[:200])
