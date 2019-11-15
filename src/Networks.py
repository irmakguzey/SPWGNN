from keras.layers import Permute,Subtract,Add,Lambda,Input,Concatenate,TimeDistributed,Activation,Dropout,dot,Reshape
import tensorflow as tf
from keras.activations import tanh,relu
from keras import optimizers
from keras import regularizers
from Blocks import *

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
        permuted_senders_rel=Permute((2,1))(sender_relations) # permutes the 1th and 2nd dimensions of sender_relations (swaps the two columns i guess)
        permuted_receiver_rel=Permute((2,1))(receiver_relations)
        # relation_info = Input(shape=(n_relations,relation_dim), name='relation_info')
        propagation = Input(shape=(n_objects,100), name='propagation')

        # Getting sender and receiver objects
        senders=dot([permuted_senders_rel,objects],axes=(2,1)) # dot == matrix multiplication
        receivers=dot([permuted_receiver_rel,objects],axes=(2,1))

        print('** CODE LOG: objects is: {}'.format(objects))
        print('** CODE LOG: sender_relations: {}'.format(sender_relations))
        print('** CODE LOG: permuted_senders_rel: {}'.format(permuted_senders_rel))
        print('** CODE LOG: senders is: {}'.format(senders))

        # TODO: Check how the senders and receivers look at the end and fix these methods accordingly
        # TODO: Check whether these are the only necessary methods for receiving attributes
        # Right now, relation has a distance feature between two objects
        # get_rel_dist = Lambda(lambda x: x[:,:,:,0:2], output_shape=(n_relations, 2))
        # # Getting specific features of objects for objectNetwork        
        # Right now the features of the object is the stabilities and their positions
        get_stability = Lambda(lambda x: x[:,:,:,3], output_shape=(n_objects, 1))
        get_obj_pos = Lambda(lambda x: x[:,:,:,0:2], output_shape=(n_objects, 2))

        # Using the same variables as @Fzaero's BRDPN depository
        if(self.set_weights):
            rm = RelationalModel((n_relations,),8+relation_dim,[150,150,150,150],self.relnet,True)
            om = ObjectModel((n_objects,),4,[100,100],self.objnet,True)
            rmp = RelationalModel((n_relations,),350,[150,150,100],self.relnetp,True)
            omp = ObjectModel((n_objects,),300,[100,102],self.objnetp,True)
        else:
            rm=RelationalModel((n_relations,),8+relation_dim,[150,150,150,150])
            om = ObjectModel((n_objects,),4,[100,100])
            
            rmp=RelationalModel((n_relations,),350,[150,150,100])
            omp = ObjectModel((n_objects,),300,[100,102])
            
            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()

        # TODO assumption is that get_attributes is unnecessary for this model
        r_pos = get_obj_pos(receivers)
        s_pos = get_obj_pos(senders)
        # Difference of positions in 
        diff_rs = Subtract()([r_pos, s_pos])

        # Getting stability of the objects
        o_stability = get_stability(objects)

        # Creating Input of Relation Network
        rel_vector_wo_prop= Concatenate()([diff_rs])
        obj_vector_wo_er=Concatenate()([get_stability(objects)])   
        
        rel_encoding=Activation('relu')(rm(rel_vector_wo_prop))
        obj_encoding=Activation('relu')(om(obj_vector_wo_er))
        rel_encoding=Dropout(0.1)(rel_encoding)
        obj_encoding=Dropout(0.1)(obj_encoding)
        prop=propagation
        prop_layer=Lambda(lambda x: x[:,:,3], output_shape=(n_objects,100))

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

        no_hand=Lambda(lambda x: x[:,1:,3], output_shape=(n_objects-1,1),name='target') # every object but the dropped one
        predicted=no_hand(x)
        model = Model(inputs=[objects,sender_relations,receiver_relations,propagation],outputs=[predicted])
        
        adam = optimizers.Adam(lr=0.0001, decay=0.0)
        model.compile(optimizer=adam, loss='mse')
        self.Nets[n_objects]=model
        return model

if __name__ == "__main__":
    # taken from Test.py from the same repository
    n_objects = 7 # 6+1
    num_of_rel_type = 1 # for now we only have the distance relation
    n_relations = n_objects*(n_objects-1)
    n_of_traj = 5

    prop_net = PropagationNetwork()

    first_model = prop_net.getModel(10)

    random_string = 'rEQqwoeU'
    with open('../data/first_model_{}_{}_{}.txt'.format(n_objects-1, n_of_traj, random_string)) as json_file:
        data = json.load(json_file)
        # This is the attribute number of an object, for this model there are 3 dimensions:
        # x position, y position, whether the objects stays stable or not
        n_object_attr_dim = 3 
        n_of_traj = len(data)
        f_lengths = [len(t[0]) for t in data]
        n_of_frame = min(f_lengths)
        # print('n_of_traj: {}, n_objects: {}, n_of_frame: {}'.format(n_of_traj, n_objects, n_of_frame))

        boxes = np.zeros((n_of_traj, n_objects, n_of_frame, n_object_attr_dim))

        # Fix the data into a numpy array
        for t in range(n_of_traj):
            for o in range(n_objects):
                for f in range(n_of_frame):
                    boxes[t][o][f][0] = data[t][o][f][0]
                    boxes[t][o][f][1] = data[t][o][f][1]

        # print('boxes: {}'.format(boxes))
        val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        # val_relation_info = np.zeros((n_of_traj, n_relations, num_of_rel_type))
        propagation = np.zeros((n_of_traj, n_objects, 100)) # TODO understand 100
        cnt = 0 # cnt will indicate the relation that we are working with
        # TODO turn this into a data structure
        relation_threshold = 1000
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    # norm function gives the root of sum of squares of every element in the given array-like
                    # inzz is a matrix with 1s and 0s indicating whether the 
                    inzz = np.linalg.norm(boxes[:,m,0,0:2] - boxes[:,j,0,0:2], axis=1) < relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt] = 1.0
                    
                    # TODO take a second look to detect whether holding the relation info is necessary or not
                    cnt += 1
                    # break
                    
            # break
        # print('val_sender_relations: {}'.format(val_sender_relations))
        # print('val_receiver_relations: {}'.format(val_receiver_relations))
        for i in range(1, n_of_frame):
            # stabilities are an array of 1s and 0s, indicating which object is going to stay stable
            stabilities = first_model.predict({'objects': boxes[:,:,i-1,:], 'sender_relations': val_sender_relations,
                                                'receiver_relations': val_receiver_relations, 'propagation': propagation})
            boxes[:,:,i,2] = stabilities[:]
            val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
            val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
            cnt = 0
            for m in range(n_objects):
                for j in range(n_objects):
                    if (m != j):
                        inzz = np.linalg.norm(boxes[:,m,i,0:2]-boxes[:,j,i,0:2], axis=1) < relation_threshold
                        val_receiver_relations[inzz, j, cnt] = 1.0
                        val_sender_relations[inzz, m, cnt] = 1.0
                        cnt += 1
            break
        
        print(boxes[:,:,0:2,2])
        predicted_stabilities = boxes[:,:,:,3]
        print(predicted_stabilities)
