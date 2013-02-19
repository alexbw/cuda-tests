from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from MouseData import MouseData
m = MouseData('mouse_mesh_low_poly3.npz')
translations = m.joint_translations

def inv(E):
    out = E.copy()
    out[:-1,:-1] = E[:-1,:-1].T
    out[:-1,-1]  = E[:-1,:-1].T.dot(-E[:-1,-1])
    return out

out = np.zeros((5,4,4))
out[:,-1,-1] = 1.
def Es(angles):
    cosines = np.cos(angles)
    sines = np.sin(angles)
    for idx in range(5):
        cx, cy, cz = cosines[idx]
        sx, sy, sz = sines[idx]
        out[idx,:-1,:] = np.array(( (cy*cz,          -cy*sz,         -sy   , translations[idx,0]),
                                    (cx*sz-sx*sy*cz, cx*cz+sx*sy*sz, -sx*cy, translations[idx,1]),
                                    (sx*sz+sy*cx*cz, sx*cz-cx*sy*sz,  cx*cy, translations[idx,2]), ))
    return out


fixed_Es = Es(m.joint_rotations)
fixed_Ms = np.empty((5,4,4))
fixed_Ms[0] = inv(fixed_Es[0])
for idx in range(1,5):
    # cumulative right-product
    fixed_Ms[idx] = fixed_Ms[idx-1].dot(inv(fixed_Es[idx]))

changed_Ms = np.empty((5,4,4))
def get_Ms(angles):
    changed_Es = Es(angles)
    changed_Ms[0] = changed_Es[0]
    for idx in range(1,5):
        # cumulative left-product
        changed_Ms[idx] = changed_Es[idx].dot(changed_Ms[idx-1])
    return [fixed_M.dot(changed_M) for fixed_M, changed_M in zip(fixed_Ms, changed_Ms)]


if __name__ == "__main__":
    # unposed stuff means M's are identities
    print "The following matrices should be all identity matrices (within machine eps)"
    M = get_Ms(np.deg2rad(m.joint_rotations))
    for iM in M:
        print iM
        assert np.allclose(iM, np.eye(4)), "Must be close to identity"

    # Now it's time to pose the mouse vertices, and view them to see that we're posing correctly.
    # We're basically overriding what happens in Skin.get_posed_vertices

    # Transform the vertices
    def pose_vertices(theMouseData, M):
        vv = np.zeros_like(theMouseData.vertices)
        for i in range(len(M)):
            A = theMouseData.joint_weights[:,i] * \
                    (np.array(np.dot(theMouseData.vertices, M[i]))).T
            vv += A.T
        return vv


    vv = pose_vertices(m, M)

    new_rotations = m.joint_rotations.copy()
    new_rotations[2,1] += 30.0
    M2 = get_Ms(np.deg2rad(new_rotations))
    vv2 = pose_vertices(m, M2)

    # Plot the mouse from the top-down
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1);
    plt.plot(vv[:,2], vv[:,0], 'o');
    plt.title("Unposed mouse")

    subplot(1,2,2);
    plot(vv2[:,2], vv2[:,0], 'o')
    title("Posed mouse, third joint 30deg right")