# Joint, RootJoint, ChildJoint
# JointChain

# TODO: constraints

import transformations as tr
import numpy as np

class Joint(object):
	"""Joint: 
	Implements a joint that will eventually be placed
	in a joint chian"""
	def __init__(self, rotation, translation, parentJoint=None, childJoint=None):
		super(Joint, self).__init__()
		self.rotation = np.copy(rotation)
		self.translation = np.copy(translation)
		self.parentJoint = parentJoint
		self.childJoint = childJoint

		self.L = None # local joint matrix. This, when multiplied into a 
					  # parent joint's, provides its world location

		self.W = None # cached world location of a joint
		self.B = None # This is the binding matrix, defined as 
					  # the initial, unperturbed, world location of the joint.
		self.Bi = None # the inverse of the bind pose matrix
		self.M = None # skinning transform matrix. This will be applied to vertices
					  # in a mesh to transform with respect to the original binding pose.

		self.calc_joint_local_matrix()

	def calc_joint_local_matrix(self):
		self.L = np.eye(4)

		# Add in rotation
		q = np.r_[0., 0., 0., 1.]
		for i in range(3):
			rot_vec = np.r_[0., 0., 0.]
			rot_vec[i] = 1.
			q = tr.quaternion_about_axis(np.radians(-self.rotation[i]), rot_vec)	
			# q = tr.quaternion_multiply(q, qi)
			Q = np.matrix(tr.quaternion_matrix(q))
			self.L = np.dot(self.L, Q)
		
		# Add in translation on the first three columns of bottom row
		self.L[3,:3] = self.translation

	def get_parent_world_matrix(self):
		if self.parentJoint != None:
			return self.parentJoint.W
		else:
			return np.eye(4)

	def calc_world_matrix(self):
		parentW = self.get_parent_world_matrix()

		# Transform the parent's world matrix to create this joint's world matrix
		self.W = np.dot(self.L, parentW)

		if self.B == None: # if we have not yet set the binding pose,
			self.B = self.W.copy() # then take this joint world matrix
									 # make it the binding pose
			self.Bi = np.linalg.inv(self.B)

		# Since we've updated the world matrix, update
		# the skinning matrix
		self.calc_skinning_matrix()

	def calc_skinning_matrix(self):
		self.M = np.dot(self.Bi,self.W)

	def get_position(self):
		return np.array(np.dot(np.r_[0., 0., 0., 1], self.W))

	def get_orientation(self):
		pass


class LinearJointChain(object):
	"""A list of joints that knows a bit about forward kinematics"""
	def __init__(self, listOfJoints=[]):
		super(LinearJointChain, self).__init__()
		self.joints = listOfJoints

	def add_joint(self, joint):
		self.joints.append(joint)
		if len(self.joints) > 1:
			self.joints[-1].parentJoint = self.joints[-2]
			self.joints[-2].childJoint = self.joints[-1]

		self.solve_forward(startAtJoint=len(self.joints)-1)

	def solve_forward(self, startAtJoint=0):
		# print range(range(startAtJoint, len(self.joints))
		for i in range(startAtJoint, len(self.joints)):
			self.joints[i].calc_joint_local_matrix()
			self.joints[i].calc_world_matrix()

	def get_joint_world_positions(self):
		return np.vstack([j.get_position() for j in self.joints])

class SkinnedMesh(object):
	"""Vertices with associated joint weights """
	def __init__(self, vertices, joint_weights, jointChain):
		super(SkinnedMesh, self).__init__()
		self.vertices = vertices
		self.joint_weights = joint_weights
		self.jointChain = jointChain

	def get_posed_vertices(self):
		# Transform the vertices
		vv = np.zeros_like(self.vertices)
		for i in range(len(self.jointChain.joints)):
			A = self.joint_weights[:,i] * \
					(np.array(np.dot(self.vertices, self.jointChain.joints[i].M))).T
			vv += A.T
		return vv


		
