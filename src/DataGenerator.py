# Most of this code is taken fron @Fzaero's BRDPM repository: https://github.com/Fzaero/BRDPN/blob/master/src/DataGenerator.py
import keras
import random
import numpy as np
import copy
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,n_objects,n_of_rel_type,number_of_frames,num_of_traj,dataset,relation_threshold,isTrain=True,batch_size=100,shuffle=True):
        # TODO: batches are created from frames. Check whether this is what you want
        'Initialization'
        self.n_objects = n_objects
        self.relation_threshold=relation_threshold
        self.batch_size = batch_size
        self.n_of_features = 6
        self.num_of_traj=num_of_traj
        self.n_of_rel_type=n_of_rel_type
        self.n_relations  = n_objects * (n_objects - 1) # number of edges in fully connected graph
        self.shuffle = shuffle
        self.number_of_frames=number_of_frames
        self.currEpoch=0
        self.data=dataset
        print('self.data.shape: {}'.format(self.data.shape))
        self.indexes = 1 + np.arange(self.number_of_frames-1)
        print('self.indexes.shape: {}'.format(self.indexes.shape))
        # TODO: change this if you want to create batches from trajectories
        for i in range(1,self.num_of_traj): # Create an index array for receiving data from each trajectory and frame
            self.indexes=np.concatenate([self.indexes,(i*self.number_of_frames +1 + np.arange(self.number_of_frames-1))])
        print('self.indexes for loop has finished')
        self.std_dev_pos = 0.05*np.std(self.data[:,:,0:2])
        print('std_dev_pos: {}'.format(self.std_dev_pos))
        self.add_gaus = 0.20
        self.propagation = np.zeros((self.batch_size, self.n_objects, 100), dtype=float)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print('in __len__')
        return int(np.floor(self.num_of_traj*(self.number_of_frames-1) / self.batch_size))//2

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print('in __getitem__')
        data_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        # Generate data
        X, y = self.__data_generation(data_indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print('in on_epoch_end')
        self.currEpoch=self.currEpoch+1
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_indexes):
        """
        """
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        print('in __data_generation')
        # Initialization
        # TODO: In my opinion data_indexes is unnecessary for now 
        # Since y and X are not dependent on the frames --> the predictions are supposed to be made in the beginning of the trajectories
        # temp_data=self.data[data_indexes,:,:].copy()
        temp_data=self.data[0,:,:].copy()
        print('temp_data.shape: {}'.format(temp_data.shape))

        # Adding Gaussian noise to that data
        # TODO: check if necessary
        # if self.add_gaus>0:
        #     for i in range(self.batch_size):
        #         for j in range(self.n_objects):
        #             if (random.random()<self.add_gaus):
        #                 temp_data[i,j,0]=temp_data[i,j,0]+np.random.normal(0, self.std_dev_pos)
        #             if (random.random()<self.add_gaus):
        #                 temp_data[i,j,1]=temp_data[i,j,1]+np.random.normal(0, self.std_dev_pos)

        if self.add_gaus>0:
            for j in range(self.n_objects):
                if (random.random()<self.add_gaus):
                    temp_data[j,0]=temp_data[0,j,0]+np.random.normal(0, self.std_dev_pos)
                if (random.random()<self.add_gaus):
                    temp_data[j,1]=temp_data[0,j,1]+np.random.normal(0, self.std_dev_pos)
                        
        # cnt = 0
        # x_receiver_relations = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);
        # x_sender_relations   = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);
        # for i in range(self.n_objects):
        #     for j in range(self.n_objects):
        #         if(i != j):
        #             inzz=np.linalg.norm(temp_data[:,i,0:2]-temp_data[:,j,0:2],axis=1)< self.relation_threshold
        #             x_receiver_relations[inzz, j, cnt] = 1.0
        #             x_sender_relations[inzz, i, cnt]   = 1.0
        #             cnt += 1

        cnt = 0
        x_receiver_relations = np.zeros((self.n_objects, self.n_relations), dtype=float)
        x_sender_relations   = np.zeros((self.n_objects, self.n_relations), dtype=float)
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if(i != j):
                    print('np.linalg.norm(temp_data[i,0:2]-temp_data[j,0:2],axis=1): {}'.format(np.linalg.norm(temp_data[i,0:2]-temp_data[j,0:2],axis=1)))
                    if sum(np.linalg.norm(temp_data[i,0:2]-temp_data[j,0:2],axis=1)) < self.relation_threshold:
                        x_receiver_relations[j, cnt] = 1.0
                        x_sender_relations[i, cnt]   = 1.0
                    cnt += 1

        x_object = temp_data 
        print('x_object: {}'.format(x_object)) # x_object = ((n_objects, n_object_attr_dim))
        # TODO: y should be an array ((n_objects))
        temp_data = self.data[:,:,:].copy()
        y = self.calculate_stability(temp_data)
        print('y: {}'.format(y))
        return {'objects': x_object,'sender_relations': x_sender_relations,\
                'receiver_relations': x_receiver_relations,\
                'propagation': self.propagation},{'target': y}

    # y = array(n_objects) - this is calculated for each trajectory
    # y indicates the stability of the objects for the current trajectory
    # data will be ((self.n_of_frame, n_objects, n_object_attr_dim))
    def calculate_stability(self, data):
        
        # Look at the last 50 frame of the data and calculate whether the position of the object is the same or not
        temp_y = np.zeros(self.n_objects)
        frame_threshold = 50 # Number of frames in the end to look for stability 
        stability_threshold = self.relation_threshold / 10
        for o in range(self.n_objects):
            for f in range(self.number_of_frames-frame_threshold, self.number_of_frames-1):
                if np.linalg.norm(data[f,o,0:2]-data[f+1,o,0:2],axis=1) < stability_threshold:
                    temp_y[o] = 1.0
                else:
                    temp_y[o] = 0.0
                    break
        return temp_y
        