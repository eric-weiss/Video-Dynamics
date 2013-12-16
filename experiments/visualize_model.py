import numpy as np
from matplotlib import pyplot as pp
import cPickle as cp
from utils import tile_raster_images
import PIL.Image

f=open('W.cpl','rb')

W=cp.load(f)
M=cp.load(f)
b=cp.load(f)

print W.shape
f.close()

wpic=tile_raster_images(W.T,(4,4),(5,4),tile_spacing=(2,2))

print np.exp(b)

pp.matshow(M)
pp.show()

image=PIL.Image.fromarray(wpic)

image.save('wpic.png')
