from keras.layers import Permute,Subtract,Add,Lambda,Input,Concatenate,TimeDistributed,Activation,Dropout,dot,Reshape
import tensorflow as tf
from keras.activations import tanh,relu
from keras import optimizers
from keras import regularizers
from Blocks import *

class PropagationNetwork:
    def __init__(self):
        self.Nets={}
        self.set_weights = False # This is for reusing the model (i guess)
    def getModel(self, n_objects, object_dim=6, relation_dim=1):
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
        relation_info= Input(shape=(n_relations,relation_dim),name='relation_info')
        propagation= Input(shape=(n_objects,100),name='propagation')

        # Getting sender and receiver objects
        senders=dot([permuted_senders_rel,objects],axes=(2,1))
        receivers=dot([permuted_receiver_rel,objects],axes=(2,1))

        # TODO: Getting specific features of objects for relationNetwork/objectNetwork
        # Since features to get is not precise yet, this is a TODO

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

        # TODO after setting the necessary attributes uncomment these lines and create the propagation network

        # r_att=get_attributes(receivers)
        # s_att=get_attributes(senders)

        # r_pos=get_pos(receivers)
        # s_pos=get_pos(senders)
        # r_vel=get_vel(receivers)
        # s_vel=get_vel(senders)

        # r_posvel = Concatenate()([r_pos,r_vel])
        # s_posvel = Concatenate()([s_pos,s_vel])
        
        # # Getting dynamic state differences.
        # dif_rs=Subtract()([r_posvel,s_posvel])
        
        # # Creating Input of Relation Network
        # rel_vector_wo_prop= Concatenate()([relation_info,dif_rs,r_att,s_att])
        # obj_vector_wo_er=Concatenate()([get_velocities(objects),get_attributes2(objects)])   
        
        # rel_encoding=Activation('relu')(rm(rel_vector_wo_prop))
        # obj_encoding=Activation('relu')(om(obj_vector_wo_er))
        # rel_encoding=Dropout(0.1)(rel_encoding)
        # obj_encoding=Dropout(0.1)(obj_encoding)
        # prop=propagation
        # prop_layer=Lambda(lambda x: x[:,:,2:], output_shape=(n_objects,100))

        # Creating the propagation
        # for _ in range(5):
        #     senders_prop=dot([permuted_senders_rel,prop],axes=(2,1))
        #     receivers_prop=dot([permuted_receiver_rel,prop],axes=(2,1))
        #     rmp_vector=Concatenate()([rel_encoding,senders_prop,receivers_prop])
        #     x = rmp(rmp_vector)
        #     effect_receivers = Activation('tanh')(dot([receiver_relations,x],axes=(2,1)))
        #     omp_vector=Concatenate()([obj_encoding,effect_receivers,prop])#
        #     x = omp(omp_vector)
        #     prop=Activation('tanh')(Add()([prop_layer(x),prop]))


        # no_hand=Lambda(lambda x: x[:,1:,:2], output_shape=(n_objects-1,2),name='target')
        # predicted=no_hand(x)
        # model = Model(inputs=[objects,sender_relations,receiver_relations,relation_info,propagation],outputs=[predicted])
        
        # adam = optimizers.Adam(lr=0.0001, decay=0.0)
        # model.compile(optimizer=adam, loss='mse')
        # self.Nets[n_objects]=model
        # return model



