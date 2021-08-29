# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import os
import numpy as np
from klepto.archives import file_archive

def generate_motion_prediction_set(root_dir, seq_length):
	DATA_PARAMS = {}
	DATA_PARAMS.update({"data_source": "MVNX", "nb_frames":seq_length, 'as_3D': True, 'data_types': ['position'],"unit_bounds": False, "path":root_dir})
	data_driver = MVNXDataDriver(DATA_PARAMS)
	data_driver.parse(frameMod=True)
	shape = data_driver.data.shape
	data = data_driver.data.reshape(-1, shape[-2], shape[-1])
	return data[:, :-1, :], data[:, 1:, :]


# Adapted from https://github.com/inria-larsen/activity-recognition-prediction-wearable/tree/master/VTSFE/data/7x10_actions
class MVNXDataDriver():

	DATA_PARAMS = {
		"path": "./data/7x10_actions/XSens/xml",
		"nb_frames": 70,
		"nb_samples_per_mov": 10,
		"mov_types": [
			"bent_fw",
			"bent_fw_strongly",
			"kicking",
			"lifting_box",
			# "standing",
			"walking",
			"window_open",
		],
		"normalization_as_3d": True,
		"relative_movement": True,
		"use_center_of_mass": True
	}

	hard_segments = [
		24, 25, 26,
		27, 28, 29,
		30, 31, 32,
		36, 37, 38,
		39, 40, 41,
		42, 43, 44,
		48, 49, 50,
		51, 52, 53,
		54, 55, 56,
		60, 61, 62,
		63, 64, 65,
		66, 67, 68
	]

	hard_joints = [
		21, 22, 23,
		24, 25, 26,
		33, 34, 35,
		36, 37, 38,
		42, 43, 44,
		45, 46, 47,
		54, 55, 56,
		57, 58, 59
	]

	joints = [
		"Pelvis/L5",
		"L5/L3",
		"L3/T12",
		"T12/T8",
		"T8/Neck",
		"Neck/Head",
		"T8/RightShoulder",
		"RightShoulder/RightUpperArm",
		"RightUpperArm/RightForeArm",
		"RightForeArm/RightHand",
		"T8/LeftShoulder",
		"LeftShoulder/LeftUpperArm",
		"LeftUpperArm/LeftForeArm",
		"LeftForeArm/LeftHand",
		"Pelvis/RightUpperLeg",
		"RightUpperLeg/RightLowerLeg",
		"RightLowerLeg/RightFoot",
		"RightFoot/RightToe",
		"Pelvis/LeftUpperLeg",
		"LeftUpperLeg/LeftLowerLeg",
		"LeftLowerLeg/LeftFoot",
		"LeftFoot/LeftToe",
	]

	segments = [
		"Pelvis",
		"L5",
		"L3",
		"T12",
		"T8",
		"Neck",
		"Head",
		"RightShoulder",
		"RightUpperArm",
		"RightForeArm",
		"RightHand",
		"LeftShoulder",
		"LeftUpperArm",
		"LeftForeArm",
		"LeftHand",
		"RightUpperLeg",
		"RightLowerLeg",
		"RightFoot",
		"RightToe",
		"LeftUpperLeg",
		"LeftLowerLeg",
		"LeftFoot",
		"LeftToe",
	]

	def __init__(self, params={}):
		self.__dict__.update(self.DATA_PARAMS, **params)
		self.scaling_factors = []
		joints = []
		for joint in self.joints:
			for dim in ["_x", "_y", "_z"]:
				joints.append(joint+dim)
		self.joints = joints
		segments = []
		for segment in self.segments:
			for dim in ["_x", "_y", "_z"]:
				segments.append(segment+dim)
		self.segments = segments


	def save_data(self, path, index, data):
		db = file_archive(path)
		db[index] = data
		db.dump()


	def read_data(self, path, index):
		db = file_archive(path)
		db.load()
		return db[index]


	def unit_bounds_rescaling(self, sample, keep_zero_mean=True):
		""" Bounds values to [-1, 1]
		"""
		self.scaling_factors = []
		rescaled_sample = np.copy(sample)
		sample_shape = rescaled_sample.shape

		if keep_zero_mean:
			max_val = np.amax(np.absolute(rescaled_sample))
			rescaled_sample /= max_val
			self.scaling_factors.append(max_val)
		else:
			# normalization to range [0,1]
			max_val = np.amax(rescaled_sample)
			min_val = np.amin(rescaled_sample)
			rescaled_sample -= min_val
			rescaled_sample /= (max_val-min_val)
			# scaling to [-1,1]
			rescaled_sample *= 2.
			rescaled_sample -= 1.
			self.scaling_factors.append((max_val, min_val))

		print("Data rescaled to unit bounds [-1, 1].")
		return rescaled_sample

	def parse(self, filenames=[], single_action=False, frameMod=True):
		""" Parses all MVNX files in folderpath, searching for values in data_types for each frame of movement.
		Each file in folderpath is a sample.
		You also may define the number of samples per movement type, as well as the number of movement types.

		data_types is a list of strings specifying the wanted data types.
		data types available :
			- position
			- orientation
			- velocity
			- acceleration
			- angularVelocity
			- angularAcceleration
			- sensorAngularVelocity
			- sensorOrientation
			- jointAngle
			- jointAngleXZY
			- centerOfMass
		"""

		ns = {"mvnx": "http://www.xsens.com/mvn/mvnx"}

		if len(self.data_types) == 1:
			if "position" in self.data_types:
				self.hard_dimensions = self.hard_segments
				self.dim_names = self.segments
			else:
				self.hard_dimensions = self.hard_joints
				self.dim_names = self.joints


		def extract(filepaths, frameMod=True):
			data = []
			labels = []

			# initialize data : first dim for mov_type, second dim for mov sample, then dims for actual data
			if len(filepaths)>1 and not single_action:
				for i in range(len(self.mov_types)):
					data.append([])
			else:
				data.append([])
			max_frame = 0
			print("Parsing MVNX files ------------")
			for filepath, label in filepaths:
				# print(filepath, label)
				tree = ET.parse(filepath)
				root = tree.getroot()

				# frames of one sample
				# frames shape = [nb_frames, n_input]
				frames = root.findall("./mvnx:subject/mvnx:frames/mvnx:frame[@type='normal']", ns)
				nb_frames = len(frames) #real number of data
				if(max_frame < nb_frames):
					max_frame = nb_frames
				if (nb_frames < self.nb_frames and frameMod):
					print(filepath+" : mvnx file has "+str(nb_frames)+" frames, which is less than "+str(self.nb_frames)+" frames. File ignored.")
					continue
				if(frameMod):
					frame_mod = nb_frames % self.nb_frames
					if frame_mod == 0:
						sampling_step = int(nb_frames / self.nb_frames)
					else:
						sampling_step = int((nb_frames-frame_mod) / self.nb_frames)
				else:
					sampling_step = 1
					self.nb_frames = nb_frames
				com = []
				sample_data = []
				step = 0
				# subsampling
				
				for i in range(self.nb_frames):# pour chaque frame
					values = []
					for data_type in self.data_types: #pour chaque data_type (dans mes tests jointAngle uniquement)
						values += frames[step].find("mvnx:"+data_type, ns).text.split() #les 66 joints au temps step
					if self.use_center_of_mass:
						# center of mass per frame and per 3D dimension.
						com.append(frames[step].find("mvnx:centerOfMass", ns).text.split())
					# sample_data shape = [self.nb_frames, n_input]
					sample_data.append(values)
					step += sampling_step
				
				sample_data = np.array(sample_data, dtype=np.float32)
				#print("shape sample data:"+str(sample_data.shape))
				if self.normalization_as_3d:
					# reshape to make 3D coordinates appear
					sample_data = sample_data.reshape([self.nb_frames, -1, 3])

					# data normalization as in article, i.e. zero mean and range in [-1, 1])

					# If we want values to be relative to the barycenter of all positions at a given frame
					if self.relative_movement:
						if self.use_center_of_mass:
							m = np.array(com, dtype=np.float32)
							for i in range(self.nb_frames):
								for d in range(3):
									sample_data[i, :, d] -= m[i, d]
						else:
							# mean of all values. One mean value per frame and per 3D dimension.
							m = np.mean(sample_data, axis=1)
							for i in range(self.nb_frames):
								for d in range(3):
									sample_data[i, :, d] -= m[i, d]
					else:
						# mean of all values. One mean value per source and per 3D dimension.
						m = np.mean(sample_data, axis=0)
						# mean of all values. One mean value per 3D dimension.
						m = np.mean(m, axis=0)
						for d in range(3):
							sample_data[:, :, d] -= m[d]
				
				labels.append(label)
				# data shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
				if len(filepaths)>1 and not single_action:
					data[self.mov_types.index(label)].append(sample_data)
				else:
					data[0].append(sample_data)
				# print(filepath+" : mvnx file parsed successfully.",sample_data.shape)
			print("------------MVNX files parsed successfully")
			if(frameMod==False):
				#todo: améliorer ici (je rempli les frame manquante par rapport a la durée max en copiant la derniere position)
				#remplissage données
				tmp_tab = np.zeros([len(data),len(data[0]),max_frame,len(data[0][0][0]),len(data[0][0][0][0])], np.float32)
				for idx in range(len(data)): # chaque type = 7
					for val in range(len(data[idx])): #chaque ex =10
						value =data[idx][val][len(data[idx][val])-1]
						for val2 in range(len(data[idx][val])):
							tmp_tab[idx][val][val2] = data[idx][val][val2]
						for val2 in range(len(data[idx][val])+1, max_frame):
							tmp_tab[idx][val][val2] = value
				data = None
				tmp_tab = np.array(tmp_tab, dtype=np.float32)
				tmp_shape = tmp_tab.shape
				data =tmp_tab.reshape(tmp_shape[0],tmp_shape[1], tmp_shape[2],tmp_shape[3]*tmp_shape[4])
				
				
				print("max frame="+ str(max_frame))
				self.nb_frames = max_frame

			else:
				data = np.array(data, dtype=np.float32)
			
			if not self.normalization_as_3d:
				# data normalization as in article, i.e. zero mean for every joint and in range [-1, 1]
				joint_means = np.mean(data, axis=0)
				joint_means = np.mean(joint_means, axis=0)
				joint_means = np.mean(joint_means, axis=0)
				for i, m in enumerate(joint_means):
					data[:, :, :, i] -= m

			# bounds values to [-1, 1]
			if self.unit_bounds:
				data = self.unit_bounds_rescaling(data)
			if(frameMod==True):
			# output data shape = [nb_mov_types, nb_samples_per_mov, nb_frames, segment_count]
				if len(filepaths)>1 and not single_action:
					data = data.reshape([len(self.mov_types), self.nb_samples_per_mov, self.nb_frames, -1])
				else:
					data = data.reshape([1, self.nb_samples_per_mov, self.nb_frames, -1])
			return data, labels

		filepaths = []
		
		samples_per_mov_count = [0]*len(self.mov_types)
		self.mov_indices = {}

		if len(filenames)>0: 
			for filename in filenames:
				if isfile(filename):
					print('isfile(filename)')
					mov_type = os.path.splitext(os.path.basename(filename))[0]
					mov_type = mov_type[:mov_type.index("-")]
					print(mov_type)
					if mov_type in self.mov_types:
						print(mov_type,'in self.mov_types')
						mov_type_index = self.mov_types.index(mov_type)
						if samples_per_mov_count[mov_type_index] < self.nb_samples_per_mov:
							print('Adding to filepaths')
							samples_per_mov_count[mov_type_index] += 1
							filepaths.append((filename, mov_type))
		else:
			for f in sorted(listdir(self.path)):
				path = join(self.path, f)
				if isfile(path):
					
					mov_type = f[:f.index("-")]
					if mov_type in self.mov_types:
						mov_type_index = self.mov_types.index(mov_type)
						if samples_per_mov_count[mov_type_index] < self.nb_samples_per_mov:
							samples_per_mov_count[mov_type_index] += 1
							filepaths.append((path, mov_type))
		print('self.mov_types:',self.mov_types)
		# print(filepaths)
		data, labels = extract(filepaths, frameMod=frameMod)
		if len(filepaths)>1 and not single_action:
			for i, mov in enumerate(self.mov_types):
				self.mov_indices[mov] = labels.index(mov)
		self.data = np.array(data, dtype=np.float32)
		self.data_labels = np.array(labels)
		print("data shape", self.data.shape)
		self.input_dim = self.data.shape[-1]
		


	def saveData(self,nbLS=69):
		
		for actionType in range(7):
			#print("saving data "+str(actionType))
			try:
			  #  print("try to create: "+"./data/observations/"+self.data_labels[(actionType*10)+1])
				os.mkdir("./data/observations/"+self.data_labels[(actionType*10)+1])
			except OSError:
			   # print("Error during creation: "+"./data/observations/"+self.data_labels[(actionType*10)+1])
				pass
			for innx in range(10):
				f = open("./data/observations/"+self.data_labels[(actionType*10)+1]+"/record"+str(innx)+".txt", "w+")
				for vb in range(0,70):
					nameString = ''
					
					for nbstring in range(nbLS-1):
						nameString += str(self.data[actionType,innx,vb,nbstring])+"\t"
					nameString +=str(self.data[actionType,innx,vb,nbLS-1])+"\n"
					f.write(nameString)
				#print("saving test "+str(innx))
				f.close()
	
	
	def split(self, nb_blocks, nb_samples, train_proportion=0, test_proportion=0, eval_proportion=0):
		if train_proportion < 0 or test_proportion < 0 or eval_proportion < 0:
			return None
		norm_sum = train_proportion + test_proportion + eval_proportion
		if norm_sum == 0:
			return None
		block_size = int(nb_samples / nb_blocks)
		nb_train_blocks = int(train_proportion * nb_blocks / norm_sum)
		nb_test_blocks = int(test_proportion * nb_blocks / norm_sum)
		nb_eval_blocks = nb_blocks - nb_train_blocks - nb_test_blocks
		return block_size, nb_train_blocks, nb_test_blocks, nb_eval_blocks


	def to_set(self, data):
		# set shape = [nb_samples, nb_frames, n_input]
		return np.concatenate(data, axis=0)


	def get_whole_data_set(self, shuffle_dataset=True):
		return self.get_data_set(range(self.nb_samples_per_mov), shuffle_dataset=shuffle_dataset, shuffle_samples=False)[0]


	def get_data_set(self, sample_indices, shuffle_samples=True, shuffle_dataset=True):
		data_copy = np.copy(self.data)
		if sample_indices is not None:
			remains_indices = []
			for index in range(self.nb_samples_per_mov):
				if index not in sample_indices:
					remains_indices.append(index)
		if shuffle_samples:
			for mov_type in data_copy:
				np.random.shuffle(mov_type)
		samples = np.reshape(data_copy[:, sample_indices], [-1, self.nb_frames, self.input_dim])
		remains = np.reshape(data_copy[:, remains_indices], [-1, self.nb_frames, self.input_dim])
		if shuffle_dataset:
			np.random.shuffle(samples)
			np.random.shuffle(remains)
		return samples, remains

