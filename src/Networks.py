from keras.layers import Permute,Subtract,Add,Lambda,Input,Concatenate,TimeDistributed,Activation,Dropout,dot,Reshape
import tensorflow as tf
from keras.activations import tanh,relu
from keras import optimizers
from keras import regularizers
from Blocks import *
from DataGenerator import *

import json
import numpy as np

class PropagationNetwork:
    def __init__(self):
        self.Nets={}
        self.set_weights = False # This is for reusing the model (i guess)
    def getModel(self, n_objects, object_dim=4, relation_dim=1): # objects dim is (in this model): width, height, x position, y position
        if n_objects in self.Nets.keys():
            return self.Nets[n_objects]
        
        n_relations = n_objects * (n_objects - 1)

        # These inputs are put in the initialization
        # Input constructor only creates a template for a layer names objects
        # objects will be full when this class is used as a model and initialized with a dict
        # with 'objects' key
        objects = Input(shape=(n_objects, object_dim), name='objects')

       # relations matrices are filled with 1 and 0s indicating whether two indiced objects have relationship or not
        sender_relations = Input(shape=(n_objects, n_relations), name='sender_relations')
        receiver_relations = Input(shape=(n_objects, n_relations), name='receiver_relations')
        permuted_senders_rel=Permute((2,1))(sender_relations) # permutes the 1th and 2nd dimensions of sender_relations
        permuted_receiver_rel=Permute((2,1))(receiver_relations)
        propagation = Input(shape=(n_objects,100), name='propagation')

        # Getting sender and receiver objects
        senders=dot([permuted_senders_rel,objects],axes=(2,1)) # dot == matrix multiplication
        receivers=dot([permuted_receiver_rel,objects],axes=(2,1))

        # We can say that it is more possible that the object is going to stay stable if it is on the ground
        get_y = Lambda(lambda x: x[:,:,1:2], output_shape=(n_objects, 1))
        get_obj_pos = Lambda(lambda x: x[:,:,0:2], output_shape=(n_objects, 2))

        # Using the same variables as @Fzaero's BRDPN depository
        if(self.set_weights):
            rm = RelationalModel((n_relations,),2,[150,150,150,150],self.relnet,True) # TODO change
            om = ObjectModel((n_objects,),1,[100,100],self.objnet,True)
            rmp = RelationalModel((n_relations,),350,[150,150,100],self.relnetp,True)
            omp = ObjectModel((n_objects,),300,[100,101],self.objnetp,True)
        else:
            rm=RelationalModel((n_relations,),2,[150,150,150,150])
            om = ObjectModel((n_objects,),1,[100,100])
            
            rmp=RelationalModel((n_relations,),350,[150,150,100])
            omp = ObjectModel((n_objects,),300,[100,101])
            
            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()

        r_pos = get_obj_pos(receivers)
        s_pos = get_obj_pos(senders)
        # Difference of positions in 
        diff_rs = Subtract()([r_pos, s_pos])
        # Getting stability of the objects
        objects_y = get_y(objects)
        # Creating Input of Relation Network
        rel_vector_wo_prop = diff_rs
        obj_vector_wo_er = objects_y
        
        rel_encoding=Activation('relu')(rm(rel_vector_wo_prop))
        obj_encoding=Activation('relu')(om(obj_vector_wo_er))
        rel_encoding=Dropout(0.1)(rel_encoding)
        obj_encoding=Dropout(0.1)(obj_encoding)
        prop=propagation
        prop_layer=Lambda(lambda x: x[:,:,1:], output_shape=(n_objects,100)) # 100 is the layer size

        # Creating the propagation
        for _ in range(5):
            senders_prop=dot([permuted_senders_rel, prop], axes=(2,1))
            receivers_prop=dot([permuted_receiver_rel, prop], axes=(2,1))
            rmp_vector=Concatenate()([rel_encoding,senders_prop,receivers_prop])
            x = rmp(rmp_vector) 
            effect_receivers = Activation('tanh')(dot([receiver_relations, x], axes=(2,1)))
            omp_vector=Concatenate()([obj_encoding, effect_receivers, prop])
            x = omp(omp_vector)
            prop=Activation('tanh')(Add()([prop_layer(x), prop]))

        sigmoid=Activation('sigmoid')
        no_hand=Lambda(lambda x: sigmoid(x[:,:,:1]), output_shape=(n_objects, 1), name='target') # make prediction for all of the objects
        print('x: {}'.format(x.shape))
        predicted=no_hand(x)
        # predicted = np.zeros(n_objects)
        print('predicted: {}'.format(predicted.shape))
        model = Model(inputs=[objects,sender_relations,receiver_relations,propagation],outputs=[predicted])
        
        adam = optimizers.Adam(lr=0.0001, decay=0.0)
        model.compile(optimizer=adam, loss='mse') # TODO binary 
        self.Nets[n_objects]=model
        return model

# boxes = ((n_of_traj, n_of_frame, n_objects, n_object_attr_dim))
def calculate_stability(boxes):
    # Look at the last 50 frame of the data and calculate whether the position of the object is the same or not
    n_of_traj = len(boxes)
    n_of_frame = len(boxes[0])
    n_objects = len(boxes[0][0])
    temp_y = np.zeros((n_of_traj, n_objects, 1)) # 1 is for making it 3 dimensional 
    frame_threshold = 50 # Number of frames in the end to look for stability 
    stability_threshold = 10 # The distance that will indicate that the corresponding object is stopping between frames
    for o in range(n_objects):
        for f in range(n_of_frame-frame_threshold, n_of_frame-1):
            inzz = np.linalg.norm(boxes[:,f,o,0:2] - boxes[:,f+1,o,0:2]) < stability_threshold
            temp_y[inzz, o, 0] = 1.0
    return temp_y

if __name__ == "__main__":
    # taken from Test.py from the same repository
    n_objects = 7 # 6+1
    object_dim = 2
    n_of_rel_type = 1 # for now we only have the distance relation
    n_relations = n_objects*(n_objects-1)
    n_of_traj = 1000

    prop_net = PropagationNetwork()

    first_model = prop_net.getModel(n_objects=n_objects, object_dim=object_dim)

    random_string = 'HbZ7clsI'
    with open('data/first_model_{}_{}_{}.txt'.format(n_objects-1, n_of_traj, random_string)) as json_file:
        data = json.load(json_file)
        n_object_attr_dim = 2 # x position, y position
        n_of_traj = len(data)
        f_lengths = [len(t[0]) for t in data]
        n_of_frame = min(f_lengths)

        boxes = np.zeros((n_of_traj, n_of_frame, n_objects, n_object_attr_dim))

        # Fix the data into a numpy array
        for t in range(n_of_traj):
            for o in range(n_objects):
                for f in range(n_of_frame):
                    boxes[t][f][o][0] = data[t][o][f][0]
                    boxes[t][f][o][1] = data[t][o][f][1]

        val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        propagation = np.zeros((n_of_traj, n_objects, 100)) # TODO understand 100
        cnt = 0 # cnt will indicate the relation that we are working with
        # TODO turn this into a data structure
        relation_threshold = 170 # Calculated according to the rectangle width and height
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    # norm function gives the root of sum of squares of every element in the given array-like
                    # inzz is a matrix with 1s and 0s indicating whether the 
                    inzz = np.linalg.norm(boxes[:,0,m,0:2] - boxes[:,0,j,0:2], axis=1) < relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt] = 1.0   
                    cnt += 1

        # TODO fit the model then
        # stabilities = first_model.predict({'objects': boxes[:,0,:,:], 'sender_relations': val_sender_relations,
        #                                         'receiver_relations': val_receiver_relations, 'propagation': propagation})
        # print(stabilities)

        print('n_of_frame: {}'.format(n_of_frame))
        print('n_of_traj: {}'.format(n_of_traj))

        # train_size = 7/10 # 70% of the data is reserved for training
        # np.random.shuffle(boxes)
        # train_len = int(n_of_traj * train_size)
        # boxes_train = boxes[0:train_len,:,:,:]
        # boxes_val = boxes[train_len:,:,:,:]

        y = calculate_stability(boxes)
        print('objects.shape: {}, sender_relations.shape: {}, receiver_relations.shape: {}, propagation.shape: {}. y.shape: {}'.format(boxes[:,0,:,:].shape,
                                                                                                                                       val_sender_relations.shape,
                                                                                                                                       val_receiver_relations.shape,
                                                                                                                                       propagation.shape,
                                                                                                                                       y.shape))

        first_model.fit({'objects': boxes[:,0,:,:], 'sender_relations': val_sender_relations, 'receiver_relations': val_receiver_relations, 'propagation': propagation},
                         {'target': y},
                         batch_size=64,
                         epochs=100,
                         validation_split=0.3,
                         shuffle=True,
                         verbose=1)

        # TODO use data generator instead of actual fit function
        # train_gen=DataGenerator(n_objects,n_of_rel_type,n_of_frame,n_of_traj,boxes_train,relation_threshold,True,64)
        # valid_gen=DataGenerator(n_objects,n_of_rel_type,n_of_frame,n_of_traj,boxes_val,relation_threshold,False,128)
        # first_model.fit_generator(generator=train_gen,
        #                           validation_data=valid_gen,
        #                           epochs=100,
        #                           use_multiprocessing=True,
        #                           workers=32,
        #                           verbose=1)




