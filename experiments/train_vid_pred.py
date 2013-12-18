import numpy as np
from matplotlib import pyplot as pp
import theano
import theano.tensor as T

from utils import tile_raster_images
import matplotlib.cm as cm

import cPickle as cp

import sys
sys.path.insert(0, '../models')

from LDmodel_pred_prop import LDmodel

import math

f=open('../data/bars.cpl','rb')
vid=cp.load(f)
f.close()

vid=np.asarray(vid,dtype='float32')
vid=vid-np.mean(vid,axis=0)

vid=1.0*vid/np.mean(np.abs(vid))

nt,nx=vid.shape
print vid.shape
ns=64
npcl=200

nsamps=20
lrate=2e-6

npred=1000

xdata=theano.shared(vid)

xvar=0.1

model=LDmodel(nx, ns, npcl,nsamps, xvar=xvar)

idx=T.lscalar()
x1=T.fvector()
x2=T.fvector()

								
updates0=model.forward_filter_step(x1)
inference_step=theano.function([idx],None,
								updates=updates0,
								givens={x1: xdata[idx,:]},
								allow_input_downcast=True)

ess=model.get_ESS()
get_ESS=theano.function([],ess)

updates1=model.resample()
resample=theano.function([],updates=updates1)

lr=T.fscalar()

nrg, updates2 = model.update_params(x1, x2, lr)
learn_step=theano.function([idx,lr],[nrg],
							updates=updates2,
							givens={x1: xdata[idx-1,:], x2: xdata[idx,:]},
							allow_input_downcast=True)

nps=T.lscalar()
sps, xps, updates3 = model.simulate_forward(nps)
predict=theano.function([nps],[sps,xps],updates=updates3,allow_input_downcast=True)




new_lrs=T.fvector()
updates5 = model.set_rel_lrates(new_lrs)
set_rel_lrates=theano.function([new_lrs],[],updates=updates5,
							allow_input_downcast=True)

e_hist=[]
ess_hist=[]
s_hist=[]
r_hist=[]
l_hist=[]
w_hist=[]
ploss_hist=[0.0]

th=[]

W=model.W.get_value()
M=model.M.get_value()
b=np.exp(model.ln_b.get_value())

pp.ion()
fig=pp.figure()
axW=fig.add_subplot(2,1,1)
wpic=tile_raster_images(W.T,(8,8),(8,8),tile_spacing=(1,1))
imgW=axW.matshow(wpic,cmap=cm.gray)
axM=fig.add_subplot(2,1,2)
imgM=axM.matshow(M)

resample_counter=0
learn_counter=0
for epoch in range(4):
	for i in range(nt-1):
		
		#normalizer,energies,ssamps,spreds,WTx=inference_step(vec)
		#normalizer,energies=inference_step(x)
		#h_samps=inference_step(i)
		inference_step(i)
		
		#pp.scatter(ssamps[:,0],ssamps[:,1],color='b')
		#pp.scatter(spreds[:,0],spreds[:,1],color='r')
		#pp.scatter(WTx[0],WTx[1],color='g')
		
		#pp.hist(energies,20)
		#pp.show()
		
		#print h_samps
		
		
		ESS=get_ESS()
		ess_hist.append(ESS)
		learn_counter+=1
		resample_counter+=1
		
		if resample_counter>0 and learn_counter>5:
			W=model.W.get_value()
			M=model.M.get_value()
			b=np.exp(model.ln_b.get_value())
			energy=learn_step(i, lrate)
			e_hist.append(energy)
			learn_counter=0
			l_hist.append(1)
			lrate=lrate*0.9999
		else:
			l_hist.append(0)
		
		if i%1000==0:	
			#print normalizer
			
			print 'Iteration ', i+nt*epoch, ' ================================'
			print 'ESS: ', ESS
			print 'Avg. delta W: ', np.mean(np.abs(model.W.get_value()-W))
			print 'Avg. delta M: ', np.mean(np.abs(model.M.get_value()-M))
			print 'b'
			print np.exp(model.ln_b.get_value())
			wpic=tile_raster_images(W.T,(8,8),(8,8),tile_spacing=(1,1))
			imgW.set_data(wpic)
			imgW.autoscale()
			imgM.set_data(M)
			imgM.autoscale()
			fig.canvas.draw()
			
		if ESS<float(npcl)/3.0:
			resample()
			resample_counter=0
			r_hist.append(1)
		else:
			r_hist.append(0)
		
		s_hist.append(model.s_now.get_value())
		w_hist.append(model.weights_now.get_value())
		
		if math.isnan(ESS):
			print '\nSAMPLING ERROR===================\n'
			break

W=model.W.get_value()
M=model.M.get_value()
b=np.exp(model.ln_b.get_value())


f=open('W.cpl','wb')
cp.dump(W,f,2)
cp.dump(M,f,2)
cp.dump(b,f,2)
f.close()

#print spred.shape
#print spred
#print xpred.shape
#print xpred
#print hsmps.shape
#print hsmps

s_hist=np.asarray(s_hist)
w_hist=np.asarray(w_hist)
ess_hist=np.asarray(ess_hist)
s_av=np.mean(s_hist,axis=1)

u=s_av[1:,:]-np.dot(s_av[:-1,:],model.M.get_value())

pp.figure(1)
pp.matshow(M)

e_hist=np.asarray(e_hist)
l_hist=np.asarray(l_hist)-.5
r_hist=np.asarray(r_hist)-.5


#pp.figure(3)
#pp.plot(e_hist)
#pp.figure(4)
#pp.plot(s_av)
#pp.plot(r_hist, 'r')
#pp.plot(l_hist, 'k')

pp.figure(5)
pp.plot(ess_hist)

#pp.figure(6)
#pp.plot(u)

#pp.figure(7)
#pp.hist(u.flatten(),100)

#pp.figure(5)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],color=zip(w_hist[:,i],np.zeros(len(w_hist)),1.0-w_hist[:,i]))

#pp.figure(6)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,1],color=zip(w_hist[:,i],np.zeros(len(w_hist)),1.0-w_hist[:,i]))
	
#pp.figure(7)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],color=zip(np.ones(len(w_hist)),np.zeros(len(w_hist)),np.zeros(len(w_hist)),w_hist[:,i]))


#pp.figure(7)
#pp.plot(spred)


#pp.figure(6)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,0],c='k',s=5)

#pp.figure(5)
#for i in range(npcl):
	#pp.scatter(range(len(s_hist)),s_hist[:,i,1],color=zip(np.zeros(len(w_hist)),np.zeros(len(w_hist)),np.ones(len(w_hist)),w_hist[:,i]))


pp.show()

