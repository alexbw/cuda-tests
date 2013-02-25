import numpy as np
import pymouse
from pylab import imshow, figure

from MouseData import MouseData
from MousePoser import MousePoser


numPasses = 2

m = MouseData(scenefile="mouse_mesh_low_poly3_2.npz")
mp = MousePoser(mouseModel=m, maxNumBlocks=10)
ja = mp.jointRotations_cpu.copy()
numMice = mp.numMicePerPass
scales = np.zeros((numMice,3), dtype='float32')
offsets = np.zeros_like(scales)
rotations = np.zeros_like(scales)

mm = pymouse.Mousemodel("/home/dattalab/hsmm-particlefilters/Test Data", 
                        n=1000,
                        image_size=(64,64))
mm.load_data()
mm.clean_data(normalize_images=False, filter_data=True)
img = mm.images[800].T[::-1,:].astype('float32')

likelihoods = np.zeros((numPasses*numMice,), dtype='float32')
posed_mice = np.zeros((numPasses*numMice, mp.resolutionY, mp.resolutionX), dtype='float32')

for i in range(numPasses):
    print "Posing mouse %d / %d" % (i*numMice, numPasses*numMice)
    ja = mp.jointRotations_cpu.copy()
    # ja[:,2,2] -= 15.0
    # ja[:,3,2] -= 15.0
    # ja[:,2,0] -= 45
    ja[:,1:,0] += np.random.normal(size=ja[:,1:,0].shape, scale=20)
    # ja[:,1:,1] += np.random.normal(size=ja[:,1:,0].shape, scale=20)
    ja[:,1:,2] += np.random.normal(size=ja[:,1:,0].shape, scale=20)
    scales[:,0] = np.abs(np.random.normal(size=(numMice,), scale=0.0025, loc=.30))
    scales[:,1] = np.abs(np.random.normal(size=(numMice,), scale=0.0025, loc=.30))
    scales[:,2] = np.abs(np.random.normal(size=(numMice,), scale=30, loc=200))
    offsets[:,0] = np.random.normal(loc=0, scale=5, size=(numMice,))
    offsets[:,1] = np.random.normal(loc=0, scale=5, size=(numMice,))
    offsets[:,2] = np.random.normal(loc=0.0, scale=3.0, size=(numMice,))
    # rotations[:,0] = np.random.normal(loc=0, scale=3, size=(numMice,))
    # rotations[:,1] = np.random.normal(loc=0, scale=0.01, size=(numMice,))


    l,p = mp.get_likelihoods(joint_angles=ja, \
                            scales=scales, \
                            offsets=offsets, \
                            rotations=rotations, \
                            real_mouse_image=img, \
                            save_poses=True)
    likelihoods[i*numMice:i*numMice+numMice] = l
    posed_mice[i*numMice:i*numMice+numMice,:,:] = p


idx = np.argsort(likelihoods)
best_imgs = np.hstack(posed_mice[idx[:5]])
best_imgs = np.hstack((img, best_imgs))
imshow(best_imgs)

mp.teardown()