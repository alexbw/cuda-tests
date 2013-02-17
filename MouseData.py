import numpy as np
import Joints

class MouseData(object):
    """docstring for MouseData"""
    def __init__(self, scenefile="mouse_mesh_low_poly3.npz"):
        super(MouseData, self).__init__()
        
        self.scenefile = scenefile
        f = np.load(self.scenefile)
        self.faceNormals = f['normals']
        v = f['vertices']
        self.vertices = np.ones((len(v),4), dtype='f')
        self.vertices[:,:3] = v
        self.vertex_idx = f['faces']
        self.num_vertices = self.vertices.shape[0]
        self.num_indices = self.vertex_idx.size
        self.joint_transforms = f['joint_transforms']
        self.joint_weights = f['joint_weights']
        self.joint_poses = f['joint_poses']
        self.joint_rotations = f['joint_rotations']
        self.joint_translations = f['joint_translations']
        self.num_joints = len(self.joint_translations)

        # Find the vertex with the maximum number of joints influencing it
        self.num_joint_influences =  (self.joint_weights>0).sum(1).max()
        self.num_bones = self.num_joints

        # Load up the joints properly into a joint chain
        jointChain = Joints.LinearJointChain()
        for i in range(self.num_bones):
            J = Joints.Joint(rotation=self.joint_rotations[i],\
                             translation=self.joint_translations[i])
            jointChain.add_joint(J)
        self.skin = Joints.SkinnedMesh(self.vertices, self.joint_weights, jointChain)
        self.joint_positions = self.skin.jointChain.get_joint_world_positions()

        # Calculate the indices of the non-zero joint weights
        # In the process, if we have vertices that have less
        # than the maximum number of joint influences,
        # we'll have to add in dummy joints that have no influence.
        # (this greatly simplifies things in the shader code)
        self.joint_idx = np.zeros((self.num_vertices, self.num_joint_influences), dtype='int')
        self.nonzero_joint_weights = np.zeros((self.num_vertices, self.num_joint_influences), dtype='float32')

        for i in range(self.num_vertices):
            idx = np.argwhere(self.skin.joint_weights[i,:] > 0).ravel()
            if len(idx) != self.num_joint_influences:
                num_to_add = self.num_joint_influences - len(idx)
                joints_to_add = np.setdiff1d(range(self.num_bones), idx)[:num_to_add]
                idx = np.hstack((idx, joints_to_add))
            self.joint_idx[i] = idx
            self.nonzero_joint_weights[i,:] = self.skin.joint_weights[i,self.joint_idx[i,:]]

        self.inverseBindingMatrices = np.array([np.array(j.Bi.copy()) for j in self.skin.jointChain.joints]).astype('float32')
        self.jointWorldMatrices = np.array([np.array(j.W.copy()) for j in self.skin.jointChain.joints]).astype('float32')
