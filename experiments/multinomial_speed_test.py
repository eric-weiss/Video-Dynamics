import numpy as np
import theano
import theano.tensor as T
import time
import cPickle as cp

from theano.tensor.shared_randomstreams import RandomStreams

from matplotlib import pyplot as pp

npcl=10

rng=RandomStreams()

npcl_tests=np.asarray([10*(2**i) for i in range(8)])
ns_tests=np.arange(4)+1

t_met=np.zeros((len(npcl_tests),len(ns_tests)))
t_the=np.zeros((len(npcl_tests),len(ns_tests)))
KL_met=np.zeros((len(npcl_tests),len(ns_tests)))
KL_the=np.zeros((len(npcl_tests),len(ns_tests)))

for j in range(len(npcl_tests)):
	for k in range(len(ns_tests)):
		
		npcl=npcl_tests[j]
		met_steps=ns_tests[k]
		
		print 'Np: ', npcl
		print 'Steps: ', met_steps
		
		ntest=200
		
		## setting up metropolis sampling ==================================
		
		init_multi_samp=theano.shared(np.asarray(np.arange(npcl),dtype='int64'))
		
		def multinomial_step(samp,w):
			
			u=rng.uniform(size=w.shape)
			i=rng.random_integers(size=w.shape, low=0, high=npcl-1)
			Wnow=w[samp]
			Wstep=w[i]
			probs=Wstep/Wnow
			out=T.switch(u<probs, i, samp)
			return out
		
		
		def sample_multinomial(initsamp,w,nsteps):
			
			samp0=initsamp
			samps, updates = theano.scan(fn=multinomial_step,
											outputs_info=[samp0],
											non_sequences=[w],
											n_steps=nsteps)
			
			return samps[-1]
		
		
		Tweights=T.fvector()
		nsteps=T.lscalar()
		
		
		Tm_samps, updates = theano.scan(fn=sample_multinomial,
										outputs_info=[None],
										non_sequences=[init_multi_samp,Tweights,nsteps],
										n_steps=ntest)
		
		sample_metropolis=theano.function([Tweights, nsteps],Tm_samps,
											allow_input_downcast=True)
		
		
		##setting up Theano sampling =======================================
		
		nummat=np.repeat(np.reshape(np.arange(npcl),(npcl,1)),npcl,axis=1)
		idx_mat=theano.shared(nummat.T)
		
		Tprobs=T.fvector()
		
		t_samp=rng.multinomial(size=Tprobs.shape,pvals=Tprobs)
		idxs=T.cast(T.sum(t_samp*idx_mat,axis=1),'int64')
		
		sample_theano=theano.function([Tprobs],idxs,allow_input_downcast=True)
		
		
		
		## Speed test
		
		weights=np.random.rand(npcl)
		probs=weights/np.sum(weights)
		
		
		
		m_samps=np.zeros((ntest,npcl))
		t_samps=np.zeros((ntest,npcl))
		
		tstart=time.clock()
		#for i in range(ntest):
		#	m_samps[i]=sample_metropolis(weights, met_steps)
		m_samps=sample_metropolis(weights,met_steps)
		mtime=time.clock()-tstart
		print 'Metropolis time: ', mtime
		
		ttime=0
		if met_steps==1:
			tstart=time.clock()
			for i in range(ntest):
				t_samps[i]=sample_theano(probs)
			ttime=time.clock()-tstart
			print 'Theano time: ', ttime
			thist=np.histogram(t_samps.flatten(),np.arange(npcl))
			thist=np.asarray(thist[0], dtype='float32')
			tprobs=thist/np.sum(thist)
			KL_t=np.sum(np.log(tprobs/tprobs)*tprobs)
			print 'KL_t: ',KL_t
			t_the[j,k]=ttime
			KL_the[j,k]=KL_t
		
		mhist=np.histogram(m_samps.flatten(),np.arange(npcl))
		
		
		mhist=np.asarray(mhist[0], dtype='float32')
		
		
		mprobs=mhist/np.sum(mhist)
		
		
		KL_m=np.sum(np.log(mprobs/tprobs)*mprobs)
		
		
		print 'KL_m: ',KL_m
		
		
		t_met[j,k]=mtime
		
		KL_met[j,k]=KL_m
		

f=open('multi_results.cpl','wb')
cp.dump(t_met,f,2)
cp.dump(t_the,f,2)
cp.dump(KL_met,f,2)
cp.dump(KL_the,f,2)
f.close()

