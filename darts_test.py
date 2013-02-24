import pymouse
from MouseData import MouseData
from MousePoser import MousePoser
import numpy as np
from pylab import imshow

mm = pymouse.Mousemodel("/home/dattalab/hsmm-particlefilters/Test Data", 
						n=1000,
						image_size=(64,64))
mm.load_data()
mm.clean_data(normalize_images=False)


m = MouseData(scenefile="mouse_mesh_low_poly3.npz")
mp = MousePoser(mouseModel=m, maxNumBlocks=30)
ja = np.tile(mp.jointRotations_cpu, (10,1))
ja[:,0] += np.random.normal(size=(ja.shape[0],), scale=30)
ja[:,2] += np.random.normal(size=(ja.shape[0],), scale=30)
p = mp.get_posed_mice(ja)

img = mm.images[35].T[::-1,:].astype('float32')
l = mp.get_likelihoods(ja, img)

idx = np.argsort(l)
best_imgs = np.hstack(p[idx[:5]])
imshow(best_imgs)

mp.teardown()