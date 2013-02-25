import pymouse
from MouseData import MouseData
from MousePoser import MousePoser
import numpy as np
from pylab import imshow

from MouseData import MouseData
m = MouseData(scenefile="mouse_mesh_low_poly3.npz")
mp = MousePoser(mouseModel=m, maxNumBlocks=30)
numPasses = 2
ja = np.tile(mp.jointRotations_cpu, (numPasses,1))
ja[:,0] += np.random.normal(size=(ja.shape[0],), scale=10)
ja[:,2] += np.random.normal(size=(ja.shape[0],), scale=10)
scales = np.ones((mp.numMicePerPass*numPasses,3), dtype='float32')
offsets = np.zeros_like(scales)
rotations = np.zeros_like(scales)


mm = pymouse.Mousemodel("/home/dattalab/hsmm-particlefilters/Test Data", 
                        n=1000,
                        image_size=(64,64))
mm.load_data()
mm.clean_data(normalize_images=False)


img = mm.images[35].T[::-1,:].astype('float32')

l,p = mp.get_likelihoods(joint_angles=ja, \
                        scales=scales, \
                        offsets=offsets, \
                        rotations=rotations, \
                        real_mouse_image=img, \
                        save_poses=True)


idx = np.argsort(l)
best_imgs = np.hstack(p[idx[:5]])
imshow(best_imgs)

mp.teardown()