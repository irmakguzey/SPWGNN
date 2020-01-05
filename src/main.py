from TowerCreator import *
from JengaBuilder import *
from Networks import *
from Blocks import *
import json
import pickle

# boxes = ((n_of_traj, n_of_frame, n_objects, n_object_attr_dim))


def calculate_stability(boxes):
	# Look at the last frame_threshold of the data and calculate whether the position of the object is the same or not
	n_of_traj = len(boxes)
	n_of_frame = len(boxes[0])
	n_objects = len(boxes[0][0])
	y = np.zeros((n_of_traj, n_objects, 1)) # 1 is for making it 3 dimensional 
	frame_threshold = n_of_frame # Number of frames in the end to look for stability 
	stability_threshold = 0.5 # The distance that will indicate that the corresponding object is stopping between frames
	for o in range(n_objects):
		for t in range(n_of_traj):
			pos_change = 0
			for f in range(n_of_frame-frame_threshold, n_of_frame-1):
				pos_change += np.linalg.norm(boxes[t,f,o,0:2]-boxes[t,f+1,o,0:2])
			if pos_change < stability_threshold:
				y[t,o,0] = 1.0
	return y
	prop_net = PropagationNetwork()
	prop_net = PropagationNetwork()
	prop_net = PropagationNetwork()
	prop_net = PropagationNetwork()

def train_gnn(n, N, file_str, jenga=False):
	# Get the data and train the model
	n_objects = n+1 # 6+1
	object_dim = 2
	if jenga:
		n_objects = n-1
		object_dim = 3
	# n_of_rel_type = 1 # for now we only have the distance relation
	n_relations = n_objects*(n_objects-1)
	n_of_traj = N

	prop_net = PropagationNetwork()
	gnn_model = prop_net.getModel(n_objects=n_objects, object_dim=object_dim)

	json_file = open(file_str)
	data = json.load(json_file)
	json_file.close()

	# n_object_attr_dim = 2 # x position, y position
	data = [d for (i,d) in enumerate(data) if len(d) != 0 and i < 5000] # TODO look into this bug, for some reason some trajectories had 0 objects in them
	n_of_traj = len(data)
	f_lengths = [len(t[0]) for t in data]
	n_of_frame = max(f_lengths)

	boxes = np.zeros((n_of_traj, n_of_frame, n_objects, object_dim))
	print('boxes.shape: {}'.format(boxes.shape))
	# Fix the data into a numpy array
	for t in range(n_of_traj):
		for o in range(n_objects):
			max_f = len(data[t][o])-1 # there are lots of difference at the number of frames between trajectories
			for f in range(n_of_frame):
				if f > max_f:
					boxes[t][f][o][0] = data[t][o][max_f][0] # when the frame of the current data is exceeded, the last position of the objects are saved 
					boxes[t][f][o][1] = data[t][o][max_f][1]
					if object_dim == 3:
						boxes[t][f][o][2] = data[t][o][max_f][2]
				else:
					boxes[t][f][o][0] = data[t][o][f][0]
					boxes[t][f][o][1] = data[t][o][f][1]
					if object_dim == 3:
						boxes[t][f][o][2] = data[t][o][f][2]


	val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
	val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
	propagation = np.zeros((n_of_traj, n_objects, 100))
	cnt = 0 # cnt will indicate the relation that we are working with

	relation_threshold = 170 # Calculated according to the rectangle width and height -- TODO
	for m in range(n_objects):
		for j in range(n_objects):
			if(m != j):
				# norm function gives the root of sum of squares of every element in the given array-like
				# inzz is a matrix with 1s and 0s indicating whether the 
				# TODO: this relation extraction might be wrong consdering the fact that 
				inzz = np.linalg.norm(boxes[:,0,m,0:2] - boxes[:,0,j,0:2], axis=1) < relation_threshold
				val_receiver_relations[inzz, j, cnt] = 1.0
				val_sender_relations[inzz, m, cnt] = 1.0   
				cnt += 1

	y = calculate_stability(boxes)
	print('objects.shape: {}, sender_relations.shape: {}, receiver_relations.shape: {}, propagation.shape: {}. y.shape: {}'.format(boxes[:,0,:,:].shape,
																																	val_sender_relations.shape,
																																	val_receiver_relations.shape,
																																	propagation.shape,
																																	y.shape))
	print('y is: {}'.format(y[0:10,:,0]))

	boxes = boxes/relation_threshold
	gnn_model.fit({'objects': boxes[:,0,:,:], 'sender_relations': val_sender_relations, 'receiver_relations': val_receiver_relations, 'propagation': propagation},
						{'target': y},
						batch_size=32,
						epochs=100,
						validation_split=0.2,
						shuffle=True,
						verbose=1)

	return gnn_model

# This script reads the saved trajectory, trains the graph neural network
# Runs in Python3
if __name__ == '__main__':
	n = 11
	N = 10000
	random_string = 'QgO1Xwla'
	file_str = 'data/jenga_model_{}_{}_{}.txt'.format(n, N, random_string)

	drop_model = train_gnn(n, N, file_str, jenga=True)
	prop_net = PropagationNetwork()
	drop_model_o = prop_net.getModel(n_objects=10, object_dim=3)
	drop_model_o.set_weights(drop_model.get_weights())
	simulator = JengaBuilder(11, 100, interactive=True, gnn_model=drop_model_o)
	simulator.run()
