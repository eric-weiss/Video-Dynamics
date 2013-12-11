import numpy as np
from matplotlib import pyplot as pp

import scipy.linalg as spla

import math

import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class LDmodel():
	
	'''
	Models discreet-time continuous data as a linear transformation
	of a linear dynamical system with sparse "noise".
	
	x: data
	s: latent variable
	u: sparse noise
	n: gaussian noise
	W: generative matrix
	M: dynamical matrix
	
	s_(t+1) = M*s_t + u
	x_t = W*s_t + n
	
	Approximate EM learning is performed via minibatched gradient 
	ascent on the log-likelihood. Inference/sampling is achieved with
	particle filtering. The proposal distribution in the particle filter
	ignores (for now) the predictive part and samples directly from the
	posterior specified by the generative part (as if the top equation
	didn't exist.)
	
	'''
	
	
	def __init__(self, nx, ns, npcl, xvar=1.0):
		
		#generative matrix
		init_W=np.asarray(np.random.randn(nx,ns)/0.1,dtype='float32')
		
		#normalize the columns of W to be unit length
		#(maybe unnecessary if sampling?)
		init_W=init_W/np.sqrt(np.sum(init_W**2,axis=0))
		
		#dynamical matrix
		init_M=np.asarray(np.eye(ns),dtype='float32')
		
		#sparsity parameters
		#parametrized as the exponent of ln_b to ensure positivity
		init_ln_b=np.asarray(np.zeros(ns),dtype='float32')
		
		self.W=theano.shared(init_W)
		self.M=theano.shared(init_M)
		self.ln_b=theano.shared(init_ln_b)
		
		#for ease of use
		self.b=T.exp(self.ln_b)
		
		#square root of covariance matrix of proposal distribution
		#initialized to the true root covariance
		init_cov_inv=np.dot(init_W.T, init_W)/(xvar**2) + 0.5*np.eye(ns)*np.exp(-2.0*init_ln_b)
		init_cov=spla.inv(init_cov_inv)
		init_C=spla.sqrtm(init_cov)
		init_C=np.asarray(np.real(init_C),dtype='float32')
		
		init_s_now=np.asarray(np.zeros((npcl,ns)),dtype='float32')
		init_weights_now=np.asarray(np.ones(npcl)/float(npcl),dtype='float32')
		
		init_s_past=np.asarray(np.zeros((npcl,ns)),dtype='float32')
		init_weights_past=np.asarray(np.ones(npcl)/float(npcl),dtype='float32')
		
		self.C=theano.shared(init_C)
		
		self.s_now=theano.shared(init_s_now)
		self.weights_now=theano.shared(init_weights_now)
		
		self.s_past=theano.shared(init_s_past)
		self.weights_past=theano.shared(init_weights_past)
		
		self.xvar=np.asarray(xvar,dtype='float32')
		
		self.nx=nx		#dimensionality of observed variables
		self.ns=ns		#dimensionality of latent variables
		self.npcl=npcl	#numer of particles in particle filter
		
		
		#for ease of use and efficient computation (these are used a lot)
		self.CCT=T.dot(self.C, self.C.T)
		self.cov_inv=T.dot(self.W.T, self.W)/(self.xvar**2) + 0.5*T.eye(self.ns)/(self.b**2)
		
		self.theano_rng = RandomStreams()
		
		self.params=				[self.W, self.M, self.ln_b]
		self.rel_lrates=np.asarray([  0.1,    1.0,     1.0]   ,dtype='float32')
		
		self.meta_params=     [self.C]
		self.meta_rel_lrates=[   1.0  ]
	
	
	def sample_proposal_s(self, s, xp):
		
		#s is npcl-by-ns
		#xp is 1-by-nx
		
		n=self.theano_rng.normal(size=T.shape(self.s_now))
		
		mean_term=T.dot(xp, self.W)/(self.xvar**2) + T.dot(s,self.M.T*0.5/(self.b**2))
		prop_mean=T.dot(mean_term, self.CCT)
		
		s_prop=prop_mean + T.dot(n, self.C)
		
		#I compute the term inside the exponent for the pdf of the proposal distrib
		prop_term=-T.sum(n**2)/2.0
		
		#return T.cast(s_prop,'float32'), T.cast(s_pred,'float32'), T.cast(prop_term,'float32'), prop_mean
		return s_prop, prop_term, prop_mean
	
	
	def forward_filter_step(self, xp):
		
		#need to sample from the proposal distribution first
		s_samps, prop_terms, prop_means = self.sample_proposal_s(self.s_now, xp)
		
		updates={}
		
		#now that we have samples from the proposal distribution, we need to reweight them
		
		recons=T.dot(self.W, s_samps.T)
		s_pred=self.get_prediction(self.s_now)
		
		x_terms=-T.sum((recons-T.reshape(xp,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2)
		s_terms=-T.sum(T.abs_((s_samps-s_pred)/self.b),axis=1)
		
		energies=x_terms+s_terms-prop_terms
		
		#to avoid exponentiating large or very small numbers, I 
		#"re-center" the reweighting factors by adding a constant, 
		#as this has no impact on the resulting new weights
		
		energies_recentered=energies-T.max(energies)
		
		alpha=T.exp(energies_recentered) #these are the reweighting factors
		
		new_weights_unnorm=self.weights_now*alpha
		normalizer=T.sum(new_weights_unnorm)
		new_weights=new_weights_unnorm/normalizer  #need to normalize new weights
		
		updates[self.s_past]=T.cast(self.s_now,'float32')
		
		updates[self.s_now]=T.cast(s_samps,'float32')
		
		updates[self.weights_past]=T.cast(self.weights_now,'float32')
		updates[self.weights_now]=T.cast(new_weights,'float32')
		
		#return normalizer, energies_recentered, s_samps, s_pred, T.dot(self.W.T,(xp-self.c)), updates
		#return normalizer, energies_recentered, updates
		#return h_samps, updates
		return updates
	
	
	def proposal_loss(self,C):
		
		#calculates how far off self.CCT is from the true posterior covariance
		CCT=T.dot(C, C.T)
		prod=T.dot(CCT, self.cov_inv)
		diff=prod-T.eye(self.ns)
		tot=T.sum(T.sum(diff**2))  #frobenius norm
		
		return tot
	
	
	def prop_update_step(self, C_now, lr):
		
		loss=self.proposal_loss(C_now)
		gr=T.grad(loss, C_now)
		return [C_now-lr*gr]
	
	
	def update_proposal_distrib(self, n_steps, lr):
		
		#does some gradient descent on self.C, so that self.CCT becomes
		#closer to the true posterior covariance
		C0=self.C
		Cs, updates = theano.scan(fn=self.prop_update_step,
									outputs_info=[C0],
									non_sequences=[lr],
									n_steps=n_steps)
		
		updates[self.C]=Cs[-1]
		
		loss=self.proposal_loss(Cs[-1])
		
		#updates={}
		#updates[self.C]=self.prop_update_step(self.C,lr)
		#loss=self.proposal_loss(self.C)
		
		return loss, updates
	
	
	def get_prediction(self, s):
		
		s_pred=T.dot(s, self.M)
		
		return s_pred
	
	
	def sample_joint(self, sp):
		
		t2_samp=self.theano_rng.multinomial(pvals=T.reshape(self.weights_now,(1,self.npcl))).T
		s2_samp=T.cast(T.sum(self.s_now*T.addbroadcast(t2_samp,1),axis=0),'float32')
		
		diffs=self.b*(s2_samp-sp)
		abs_term=T.sum(T.abs_(diffs),axis=1)
		alpha=T.exp(-abs_term)
		probs_unnorm=self.weights_past*alpha
		probs=probs_unnorm/T.sum(probs_unnorm)
		
		t1_samp=self.theano_rng.multinomial(pvals=T.reshape(probs,(1,self.npcl))).T
		s1_samp=T.cast(T.sum(self.s_past*T.addbroadcast(t1_samp,1),axis=0),'float32')
		
		return [s1_samp, s2_samp]
	
	
	def update_params(self, x1, x2, n_samps, lrate):
		
		#this function samples from the joint posterior and performs
		# a step of gradient ascent on the log-likelihood
		
		sp=self.get_prediction(self.s_past)
									
		#sp should be np by ns
		
		[s1_samps, s2_samps], updates = theano.scan(fn=self.sample_joint,
									outputs_info=[None, None],
									non_sequences=[sp],
									n_steps=n_samps)
		
		
		
		x2_recons=T.dot(self.W, s2_samps.T)
		
		s_pred = self.get_prediction(s1_samps)
		
		sterm=-T.mean(T.sum(T.abs_((s2_samps-s_pred)/self.b),axis=1)) - T.sum(T.log(self.b))
		
		#xterm1=-T.mean(T.sum((x1_recons-T.reshape(x1,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2))
		xterm2=-T.mean(T.sum((x2_recons-T.reshape(x2,(self.nx,1)))**2,axis=0)/(2.0*self.xvar**2))
		
		#energy = hterm1 + xterm1 + hterm2 + xterm2 + sterm -T.sum(T.sum(self.A**2))
		#energy = hterm1 + xterm2 + sterm 
		energy = xterm2 + sterm 
		
		gparams=T.grad(energy, self.params, consider_constant=[s1_samps, s2_samps])
		
		# constructs the update dictionary
		for gparam, param, rel_lr in zip(gparams, self.params, self.rel_lrates):
			#gnat=T.dot(param, T.dot(param.T,param))
			if param==self.M:
				#I do this so the derivative of M doesn't depend on the sparsity parameters
				updates[param] = T.cast(param + gparam*T.reshape(self.b,(1,self.ns))*lrate*rel_lr,'float32')
			elif param==self.b:
				updates[param] = T.cast(param + gparam*T.reshape(1.0/self.b,(1,self.ns))*lrate*rel_lr,'float32')
			else:
				updates[param] = T.cast(param + gparam*lrate*rel_lr,'float32')
		
		return energy, updates
		
	
	def get_ESS(self):
		
		return 1.0/T.sum(self.weights_now**2)
	
	
	def resample_step(self):
		
		idx=self.theano_rng.multinomial(pvals=T.reshape(self.weights_now,(1,self.npcl))).T
		s_samp=T.sum(self.s_now*T.addbroadcast(idx,1),axis=0)
		
		return T.cast(s_samp,'float32')
	
	
	def resample(self):
		
		s_samps, updates = theano.scan(fn=self.resample_step,
												outputs_info=[None],
												n_steps=self.npcl)
		
		updates[self.s_now]=T.cast(s_samps,'float32')
		updates[self.weights_now]=T.cast(T.ones_like(self.weights_now)/T.cast(self.npcl,'float32'),'float32') #dtype paranoia
		
		return updates
	
	
	def simulate_step(self, s):
		
		s=T.reshape(s,(1,self.ns))
		
		sp=self.get_prediction(s)
		
		xp=T.dot(self.W, sp.T)
		
		return T.cast(sp,'float32'), T.cast(xp,'float32')
		
	
	def simulate_forward(self, n_steps):
		
		
		s0=T.sum(self.s_now*T.reshape(self.weights_now,(self.npcl,1)),axis=0)
		s0=T.reshape(s0,(1,self.ns))
		[sp, xp], updates = theano.scan(fn=self.simulate_step,
										outputs_info=[s0, None],
										n_steps=n_steps)
		
		return sp, xp, updates



