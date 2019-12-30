import random
import math
import time
import json
import string
import numpy as np

import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse

import pymunk
from pymunk import Vec2d
import pymunk.pyglet_util

# This class creates towers with objects of the same size 
class TowerCreator(pyglet.window.Window):
    def __init__(self, n, N=0, self_run=False, predict_stability=False, demolish=False, jenga=False, gnn_model=None):
        self.n = n # Number of rectangles
        self.N = N # Number of iterations
        self.self_run = self_run
        self.predict_stability = predict_stability
        self.predicted_stability = False
        self.demolish = demolish
        self.jenga = jenga
        self.removed_object = False # Check whether the random object to remove in jenga model is removed or not
        self.gnn_model = gnn_model
        self.stabilities = []
        # Success in percentage - used both in remove_to_demolish success and predict_stability success
        # If it is used in demolish success it holds the percentage of the objects that are demolished
        # If it is used in predict success it holds the percentage of correct predicts 
        self.success = []

        self.window_width = 1500
        self.window_height = 800

        self.event_loop = pyglet.app.EventLoop()
        pyglet.window.Window.__init__(self, self.window_width,
                                        self.window_height, vsync=False)

        self.set_caption('Tower Creator')

        self.fps_display = pyglet.window.FPSDisplay(self)

        self.iteration_text = pyglet.text.Label('Number of iterations is: {}'.format(self.N),
                            font_size=10,
                            x=10, y=400)
        self.rectangle_text = pyglet.text.Label('Number of rectangles is: {}'.format(self.n),
                            font_size=10,
                            x=10, y=450)

        self.rect_width = 150 # set size for rectangles
        self.rect_height = 80
        self.relation_threshold = math.sqrt(self.rect_width ** 2 + self.rect_height ** 2)
        self.max_space_rects = 50

        self.bottom_edge = 70
        self.left_most = 500 # left_most point to put rectangle to
        self.right_most = self.window_width - self.left_most

        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.draw_options.collision_point_color = (10, 20, 30, 40)

        self.trajectories = []

        if self.self_run:
            if self.predict_stability:
                self.predict_and_calculate_success()
            elif self.demolish:
                self.demolish_and_calculate_success()
            else:
                self.run_and_take_trajectory()
        else:
            self.create_world() # Creates a world with randomly put n rectangles

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/500.0)
        self.event_loop.run()

    def run_and_take_trajectory(self):
        for i in range(self.N):
            pyglet.clock.schedule_once(self.callback, i, callback_type='create_world')
            if self.jenga:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='remove_object')
            else:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='drop_object')

        
        pyglet.clock.schedule_once(self.callback, self.N, callback_type='save_trajectories')
        pyglet.clock.schedule_once(self.event_loop.exit, self.N+1)
        print('*** scheduling over')  

    def predict_and_calculate_success(self):
        for i in range(self.N):
            pyglet.clock.schedule_once(self.callback, i, callback_type='create_world')
            if self.jenga:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='remove_object') # The program will calculate stability after this since self.predict_stability is True for sure
            else:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='drop_object')
            pyglet.clock.schedule_once(self.callback, i+0.8, callback_type='calculate_predict_success') # Calculate actual stability and compare it with the predicted one
        
        pyglet.clock.schedule_once(self.callback, self.N, callback_type='print_success')
        pyglet.clock.schedule_once(self.event_loop.exit, self.N+1)

    # This method runs remove to demolish and calculates how much of the objects were demolished
    def demolish_and_calculate_success(self):
        for i in range(self.N):
            pyglet.clock.schedule_once(self.callback, i, callback_type='create_world')
            if self.jenga:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='remove_to_demolish') # The program will calculate stability after this since self.predict_stability is True for sure
            else:
                pyglet.clock.schedule_once(self.callback, i+0.2, callback_type='drop_to_demolish')
            pyglet.clock.schedule_once(self.callback, i+0.8, callback_type='calculate_demolish_success') # Calculate actual stability and compare it with the predicted one
        
        pyglet.clock.schedule_once(self.callback, self.N, callback_type='print_success')
        pyglet.clock.schedule_once(self.event_loop.exit, self.N+1)

    def callback(self, dt, callback_type):
        if callback_type == 'create_world':
            self.create_world()
        elif callback_type == 'drop_object':
            self.drop_object()
        elif callback_type == 'remove_object':
            self.remove_object()
        elif callback_type == 'save_trajectories':
            self.save_trajectories()
        elif callback_type == 'calculate_predict_success':
            self.success.append(self.calculate_predict_success())
        elif callback_type == 'remove_to_demolish':
            self.remove_to_demolish()
        elif callback_type == 'drop_to_demolish':
            self.drop_to_demolish()
        elif callback_type == 'calculate_demolish_success':
            self.success.append(self.calculate_demolish_success())
        elif callback_type == 'print_success':
            print('self.success: {}'.format(self.success))
            print('average success is : {}'.format(sum(self.success) / len(self.success)))

    def save_trajectories(self):
        # create random endix to the file name
        letters_and_digits = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(letters_and_digits) for i in range(8))
        # dump the trajectories into a file
        if self.jenga:
            file_name = 'data/jenga_model_{}_{}_{}.txt'.format(self.n, self.N, random_string)
        else:
            file_name = 'data/second_model_{}_{}_{}.txt'.format(self.n, self.N, random_string)
        with open(file_name, 'w+') as outfile:
            json.dump(self.trajectories, outfile)

    def create_world(self):
        print('in create_world')
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0., -900.)
        self.space.sleep_time_threshold = 0.9
        self.boxes = [] # 2d array box instances correspondingly for each layer
        self.dropped_object = None
        self.dropped_trajectory = [[],[]] # First element represents x trajectory, second: y
        self.predicted_stability = False
        self.removed_object = False

        # Create the ground line
        ground_line = pymunk.Segment(self.space.static_body, Vec2d(20, self.bottom_edge), Vec2d(self.window_width-20, self.bottom_edge), 1)
        ground_line.friction = 0.9
        self.space.add(ground_line)

        # Setting up the orientation of the center of the mass
        # At some trajectories we want the boxes to build up towards right, at some of them towards left
        # left < 0.5, right > 0.5
        self.orientation = random.random() > 0.5
        print('self.orientation: {}'.format(self.orientation))

        # Setting up the number of boxes in each layer
        layers = [random.randint(1, math.floor(self.n/2))]
        n = self.n - layers[0]
        j = 1
        while n > 0:
            if layers[j-1] == 1:
                # If there is only one box then the boxes on top does not stay still with half of their center of mass staying on balance
                r = 1
            else:
                # In order to decrease the chance of having 1 the data is manipulated
                # When a layer has one box only then the rest of the tower has one box at every layer as well that is why we want the system to
                # avoid giving 1 to a layer as much as possible
                r = random.randint(1, min(layers[j-1], n))
                i = 0
                while r == 1 and n != 1 and i < 3:
                    r = random.randint(1, min(layers[j-1], n))
                    i += 1

            layers.append(r)
            n -= r
            j += 1
        print('number of boxes in each layer is: {}'.format(layers))

        # Put the boxes on top of each other
        for (layer_num, layer_size) in enumerate(layers):
            self.boxes.append([])
            middle_x = self.get_middle(layer_num)
            self.put_boxes(layer_num, layer_size, middle_x)

        # # With this model putting boxes and setting up number of boxes happens in parallel
        # n = self.n
        # layer_num = -1 # Represents the index of the current layer
        # while n > 0:
        #     layer_num += 1
        #     self.boxes.append([])
        #     right_edge, left_edge = self.get_right_left_edge(layer_num-1)

        #     if right_edge == left_edge: # The layer below has one element
        #         x_pos = random.randint(int(left_edge-self.rect_width/3), int(left_edge+self.rect_width/3))
        #         self.put_box(layer_num=layer_num,
        #                      x_pos=x_pos,
        #                      y_pos=self.bottom_edge +int(self.rect_height/2) + self.rect_height * layer_num,
        #                      rect_width=self.rect_width)
        #         n -= 1
        #         continue

        #     # Start from the left_edge and put boxes towards right edge
        #     # Stop putting boxes if the number of objects required is reached
        #     # Move on to the next layer if the right edge is reached
        #     # left_edge -= (layer_num > 0) * int(self.max_space_rects/2)
        #     left_edge += (layer_num > 0) * random.randint(0, self.max_space_rects)
        #     # space_between = random.randint(0, self.max_space_rects)
        #     # left_edge += space_between
        #     while left_edge - self.rect_width/3 < right_edge and n > 0:
        #         self.put_box(layer_num=layer_num,
        #                      x_pos=left_edge - self.rect_width/3,
        #                      y_pos=self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num,
        #                      rect_width=self.rect_width)
        #         n -= 1
        #         space_between = random.randint(0, self.max_space_rects)
        #         left_edge += space_between # Put a space between rectangles
        #         left_edge += self.rect_width

        # NOTE: flat_boxes only have the boxes in the beginning
        # but self.boxes consists of the dropped_object as well at the end
        self.flat_boxes = []
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[i])):
                self.flat_boxes.append(self.boxes[i][j])
        print('len(flat_boxes): {}'.format(len(self.flat_boxes)))
        print('len(trajectories): {}'.format(len(self.trajectories)))
        if len(self.flat_boxes) == self.n:
            self.trajectories.append([])

    def put_box (self, layer_num, x_pos, y_pos, rect_width):
        mass = 50.0
        moment = pymunk.moment_for_box(mass, (rect_width, self.rect_height))
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(x_pos, y_pos)
        shape = pymunk.Poly.create_box(body, (rect_width, self.rect_height))
        shape.friction = 0.3
        self.space.add(body, shape)
        self.boxes[layer_num].append(shape)

    # Return the rightmost/leftmost objects' middle positions of the given layer
    # def get_right_left_edge(self, layer_num):
    #     if layer_num == -1: # Means that this method is called for the first layer
    #         return self.right_most, self.left_most
    #     right_edge = -10000 # The window width cannot be 10000 pixels for sure
    #     left_edge = 10000
    #     for box in self.boxes[layer_num]:
    #         if box.body.position[0] > right_edge: 
    #             right_edge = box.body.position[0]
    #         if box.body.position[0] < left_edge:
    #             left_edge = box.body.position[0]
    #     return right_edge, left_edge 

    def create_pos_for_boxes(self, layer_num, layer_size, index_in_layer, middle_x, to_drop=False):
        box_variation = int(self.rect_width * 0.3)
        if to_drop:
            box_variation = int(self.rect_width * 0.5)
        mean_range = self.rect_width + 2 * box_variation
        box_mean = middle_x + ( ((-1) ** index_in_layer) * math.floor((index_in_layer+1)/2) * mean_range ) 

        if layer_num > 0 and layer_size == 1:
            right_edge, left_edge = self.get_right_left_edge(layer_num-1) 
            # to make the beginning of the towers a little more stable instad of going halft left half right, we go 0.3 left, 0.3 right...
            x_pos = random.randint(int(left_edge)+int(self.rect_width * 0.2), int(right_edge)-int(self.rect_width * 0.2))
        else: 
            if layer_size % 2 == 0:
                x_pos = random.randint(box_mean - (1 - self.orientation) * box_variation, box_mean + self.orientation * box_variation) + int(mean_range / 2)
            else:
                x_pos = random.randint(box_mean - (1 - self.orientation) * box_variation, box_mean + self.orientation * box_variation)

        y_pos = self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num

        return x_pos, y_pos

    # This method puts necessary number of boxes to the given layer by starting putting the box in the middle to the middle position
    def put_boxes(self, layer_num, layer_size, middle_x, to_drop=False):
        for i in range(layer_size):
            x_pos, y_pos = self.create_pos_for_boxes(layer_num=layer_num, layer_size=layer_size, index_in_layer=i, middle_x=middle_x, to_drop=to_drop)
            # Move the box a little towards the orientation
            # If orientation is 1, random part will be added (moving x towards left)
            # otherwise random part will be removed which will make x move towards right
            mass = 50.0
            moment = pymunk.moment_for_box(mass, (self.rect_width, self.rect_height))
            body = pymunk.Body(mass, moment)
            body.position = Vec2d(x_pos, y_pos)
            shape = pymunk.Poly.create_box(body, (self.rect_width, self.rect_height))

            # This block checks whether the tower getting built is stable or not 
            if not self.is_box_stable(layer_num, shape):
                print('in not self.is_box_stable')
                self.orientation = not self.orientation
                x_pos, y_pos = self.create_pos_for_boxes(layer_num=layer_num, layer_size=layer_size, index_in_layer=i, middle_x=middle_x, to_drop=to_drop)
                body = pymunk.Body(mass, moment)
                body.position = Vec2d(x_pos, y_pos)
                shape = pymunk.Poly.create_box(body, (self.rect_width, self.rect_height))  

            shape.friction = 0.3
            self.space.add(body, shape)
            self.boxes[layer_num].append(shape)

    # This method returns the middle x position of the given layer 
    # Middle x is calculated as the middle of x position of left side of the rectangle on the left most position
    # and right side of the rectangle on the right most position at the layer below
    def get_middle(self, layer_num):
        if layer_num == 0:
            return self.window_width / 2
        right_edge, left_edge = self.get_right_left_edge(layer_num-1)
        return int((left_edge + right_edge) / 2)

    # returns the right and left edge of the given layer
    def get_right_left_edge(self, layer_num):
        right_edge = -10000 # The window width cannot be 10000 pixels for sure
        left_edge = 10000
        for box in self.boxes[layer_num]:
            if box.body.position[0] > right_edge: 
                right_edge = box.body.position[0]
            if box.body.position[0] < left_edge:
                left_edge = box.body.position[0]
        return right_edge+int(self.rect_width/2), left_edge-int(self.rect_width/2)

    # Returns the x position of the current center of mass of self.boxes
    # TODO: these methods are not used for now, check whether you need them or not
    def get_center_of_mass_x(self):
        center_of_mass = 0
        num_of_boxes = 0
        for j in range(len(self.boxes)):
            for i in range(len(self.boxes[j])):
                num_of_boxes += 1

        for j in range(len(self.boxes)):
            for box in self.boxes[j]:
                center_of_mass += int(box.body.position[0] / num_of_boxes)

        return center_of_mass

    # This method returns True or False indicating whether the given x will cause the system's center of mass get out of the system's grounding boxes
    # TODO: this method is not being used, check whether you need it or not
    def is_box_stable(self, layer_num, box):
        if layer_num == 0:
            return True

        self.boxes[layer_num].append(box)
        com = self.get_center_of_mass_x()
        self.boxes[layer_num].pop() # remove the box

        # Calculate the left most and right most points which when the center of mass is between these two values the system is stable
        right_edge, left_edge = self.get_right_left_edge(0)

        return (com >= left_edge and com <= right_edge)

    def drop_object(self):
        layer_num = len(self.boxes)
        self.boxes.append([])
        middle_x = self.get_middle(layer_num)
        # When the box is put to drop the variation increases
        self.put_boxes(layer_num=layer_num, layer_size=1, middle_x=middle_x, to_drop=True)
        # right_edge, left_edge = self.get_right_left_edge(layer_num-1)
        # x_pos = random.randint(int(left_edge - self.rect_width/3), int(right_edge + self.rect_width/3))
        # self.put_box(layer_num=layer_num,
        #              x_pos=x_pos,
        #              y_pos=self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num,
        #              rect_width=self.rect_width)
        self.dropped_object = self.boxes[layer_num][0]

    # Create a random position to drop the object 
    # Add the position of the object to the trajectories (since predict stability works on the trajectories)
    # If one of them demolishes actually drop the object
    def drop_to_demolish(self):
        layer_num = len(self.boxes)
        self.boxes.append([])
        middle_x = self.get_middle(layer_num)
        
        # Add the positions to trajectories, call predict_stabilities, if the stabilities are mostly lower than 0.5, put the box there
        # Otherwise don't put the box keep creating positions
        if (len(self.trajectories[-1]) == 0):
            for _ in range(self.n+1):
                self.trajectories[-1].append([])
            
        self.trajectories[-1][0].append([0, 0]) # [0,0] gets changed as x_pos and y_pos is created below
        for i,box in enumerate(self.flat_boxes):
            self.trajectories[-1][i+1].append([box.body.position[0],box.body.position[1]])

        num_of_predicts = 100
        pos_stability = np.zeros((num_of_predicts, 3)) # 3: [stability_sum, x_pos, y_pos]

        for i in range(num_of_predicts):
            stability_sum = 0
            x_pos, y_pos = self.create_pos_for_boxes(layer_num=layer_num, layer_size=1, index_in_layer=0, middle_x=middle_x, to_drop=True)
            self.trajectories[-1][0][0] = [x_pos, y_pos]
            self.predict_stabilities()
            for s in self.stabilities[0]:
                stability_sum += s[0]
            pos_stability[i] = [stability_sum, x_pos, y_pos]
            # print('pos_stability[i]: {}'.format(pos_stability[i]))

        self.trajectories[-1] = []
        # print('stabilities after for in drop_to_demolish: {}'.format(pos_stability[:,0]))
        index_min = np.argmin(pos_stability[:,0])
        x_pos, y_pos = pos_stability[index_min, 1:]
        # print('pos_stability with minimum stability is: {}, and x_pos,y_pos found: {},{}'.format(pos_stability[index_min], x_pos, y_pos))

        # Put the object with minimum stability
        mass = 50.0
        moment = pymunk.moment_for_box(mass, (self.rect_width, self.rect_height))
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(x_pos, y_pos)
        shape = pymunk.Poly.create_box(body, (self.rect_width, self.rect_height))
        shape.friction = 0.3
        self.space.add(body, shape)
        self.boxes[layer_num].append(shape)
        self.dropped_object = self.boxes[layer_num][0]

    # This method removes one random object from the system
    def remove_object(self):
        # print('before removal: self.flat_boxes: {}, self.boxes: {}'.format(self.flat_boxes, self.boxes))
        random_index = random.randint(0, self.n-1) # get a random index to remove
        random_object = self.flat_boxes[random_index] # get the corresponding random object
        self.flat_boxes.remove(random_object) # remove that object from self.flat_boxes
        for layer in self.boxes: # remove the object from self.boxes
            if random_object in layer:
                layer.remove(random_object)
                if len(layer) == 0:
                    self.boxes.remove(layer)
        # print('after removal: self.flat_boxes: {}, self.boxes: {}'.format(self.flat_boxes, self.boxes))
        self.space.remove(random_object) # remove the object from the space
        self.removed_object = True

    # It tries to remove each of the objects and choose the one that will demolish the system
    def remove_to_demolish(self):
        if (len(self.trajectories[-1]) == 0):
            for _ in range(self.n-1):
                self.trajectories[-1].append([])

        # box_stabilities[remove_index] = stability: holds the stability of the system when the box in remove_index is removed
        box_stabilities = np.zeros(self.n)
        for remove_index in range(self.n):
            stability_sum = 0
            for box_index,box in enumerate(self.flat_boxes): # put the positions of the objects properly
                if box_index < remove_index:
                    self.trajectories[-1][box_index].append([box.body.position[0],box.body.position[1]])
                elif box_index > remove_index:
                    self.trajectories[-1][box_index-1].append([box.body.position[0],box.body.position[1]])

            # predict stabilities for the box in remove_index removed
            self.predict_stabilities()

            for s in self.stabilities[0]:
                stability_sum += s[0]
            box_stabilities[remove_index] = stability_sum

        self.trajectories[-1] = []
        print('box_stabilities: {}'.format(box_stabilities))
        remove_index = np.argmin(box_stabilities)
        remove_object = self.flat_boxes[remove_index]
        self.flat_boxes.remove(remove_object) # remove that object from self.flat_boxes
        for layer in self.boxes: # remove the object from self.boxes
            if remove_object in layer:
                layer.remove(remove_object)
                if len(layer) == 0:
                    self.boxes.remove(layer)
        self.space.remove(remove_object) # remove the object from the space
        self.removed_object = True

    # Looks at the trajectory and calculates the stabilities of each object
    # trajectory = self.trajectories[i] == (n_of_objects, n_of_frame, (x,y))
    def calculate_stability(self, trajectory_index):
        print('trajectories[-1] in calculate_stability: {}'.format(self.trajectories[-1]))
        n_of_frame = len(self.trajectories[trajectory_index][0]) # number of frame of the zeroth object
        n_of_objects = len(self.trajectories[trajectory_index]) # self.n+1

        # Reverse trajectory into a numpy array
        trajectory = np.zeros((n_of_objects, n_of_frame, 2))
        # Fix the data into a numpy array
        for o in range(n_of_objects):
            for f in range(n_of_frame):
                trajectory[o][f][0] = self.trajectories[trajectory_index][o][f][0]
                trajectory[o][f][1] = self.trajectories[trajectory_index][o][f][1]

        # Look at the last frame_threshold of the data and calculate whether the position of the object is the same or not
        stabilities = np.zeros((n_of_objects, 1)) # 1 for making it two dimensional (looks same as the predicted stability as well)
        stability_threshold = 0.5 # The distance that will indicate that the corresponding object is stopping between frames
        for o in range(n_of_objects):
            pos_change = 0
            for f in range(n_of_frame-1):
                pos_change += np.linalg.norm(trajectory[o,f,0:2]-trajectory[o,f+1,0:2]) # take difference between two different frames of object
            if pos_change < stability_threshold:
                stabilities[o,0] = 1.0

        return stabilities

    # This function returns an array (n_objects), indicating the stability of each object
    # This is called at the of the drop_object
    def predict_stabilities(self):
        n_of_traj = 1
        n_objects = self.n+1 # including the dropped object
        if self.jenga:
            n_objects = self.n-1
        n_relations = n_objects * (n_objects - 1)
        n_object_attr_dim = 2

        boxes = np.zeros((1, n_objects, n_object_attr_dim)) # 1 for 1 trajectory
        for o in range(n_objects):
            boxes[0,o,0] = self.trajectories[-1][o][0][0] / 170.0 # 170 is for relation threshold in main.py
            boxes[0,o,1] = self.trajectories[-1][o][0][1] / 170.0

        val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        propagation = np.zeros((n_of_traj, n_objects, 100))
        cnt = 0 # cnt will indicate the relation that we are working with
        # TODO turn this into a data structure
        for m in range(n_objects):
            for j in range(n_objects):
                if(m != j):
                    # norm function gives the root of sum of squares of every element in the given array-like
                    # inzz is a matrix with 1s and 0s indicating whether the 
                    inzz = np.linalg.norm(boxes[:,m,0:2] - boxes[:,j,0:2], axis=1) < self.relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt] = 1.0   
                    cnt += 1

        self.stabilities = self.gnn_model.predict({'objects': boxes[0:1,:,:], 'sender_relations': val_sender_relations,
                                                    'receiver_relations': val_receiver_relations, 'propagation': propagation})

    # Compares the actual stability of each object and the predicted stability
    # And checks how much of them was correct
    def calculate_predict_success(self):
        calculated_stabilities = self.calculate_stability(-1)
        predicted_stabilities = self.stabilities
        success = 0
        # print('calculated_stabilities: {}, predicted_stabilities: {}'.format(calculated_stabilities, predicted_stabilities))
        stability_length = len(calculated_stabilities)
        for i in range(stability_length):
            c = calculated_stabilities[i][0]
            s = predicted_stabilities[0][i][0]
            # print('s: {}, c: {}, (s > 0.5): {}'.format(s, c, s>0.5))
            success += ((s > 0.5) == c) # if ((s>0.5) == c) then success += 1

        print('success calculated is: {}%'.format(success / stability_length * 100))
        return success / stability_length * 100

    # Calculates how much of the objects are demolished
    def calculate_demolish_success(self):
        calculated_stabilities = self.calculate_stability(-1)
        stability_length = len(calculated_stabilities)
        success = 0
        
        for i in range(stability_length):
            success += (calculated_stabilities[i][0] == 0)

        return success / stability_length * 100


    def update(self, dt):
        step_dt = 1/250.
        x = 0
        while x < dt:
            x += step_dt
            self.space.step(step_dt)

        if self.jenga and self.removed_object:
            if len(self.trajectories[-1]) == 0:
                for _ in range(self.n-1):
                    self.trajectories[-1].append([])
            for i,box in enumerate(self.flat_boxes): # the object is also removed from flat_boxes
                self.trajectories[-1][i].append([box.body.position[0], box.body.position[1]])

        if not self.dropped_object == None:
            if len(self.trajectories[-1]) == 0:
                for _ in range(self.n+1):
                    self.trajectories[-1].append([])
            self.trajectories[-1][0].append([self.dropped_object.body.position[0],self.dropped_object.body.position[1]])
            for i,box in enumerate(self.flat_boxes):
                self.trajectories[-1][i+1].append([box.body.position[0],box.body.position[1]])

        if self.predict_stability and not self.predicted_stability:
            if (self.jenga and self.removed_object) or (not self.dropped_object == None):
                self.predict_stabilities()
                self.predicted_stability = True
                # self.calculate_predict_success()

    def on_draw(self):
        self.clear()
        self.iteration_text.draw()
        self.rectangle_text.draw()
        self.fps_display.draw()
        
        self.space.debug_draw(self.draw_options)
        
        # Draw lines between boxes with relationship between
        for (i,box_a) in enumerate(self.flat_boxes):
            glBegin(GL_LINES)
            for box_b in self.flat_boxes:
                if self.there_is_relation(box_a, box_b):
                    p1 = Vec2d(box_a.body.position[0], box_a.body.position[1])
                    p2 = Vec2d(box_b.body.position[0], box_b.body.position[1])
                    glVertex2f(p1.x, p1.y)
                    glVertex2f(p2.x, p2.y)

            if not self.dropped_object == None:
                if self.there_is_relation(box_a, self.dropped_object):
                    p1 = Vec2d(box_a.body.position[0], box_a.body.position[1])
                    p2 = Vec2d(self.dropped_object.body.position[0], self.dropped_object.body.position[1])
                    glVertex2f(p1.x, p1.y)
                    glVertex2f(p2.x, p2.y)
            glEnd()
        
        # Draw black boxes to the center of the boxes that are predicted to stay stable
        glPointSize(50)
        glBegin(GL_POINTS)
        glColor3i(255, 0, 0)
        if self.predicted_stability:
            if self.jenga:
                for (i, box) in enumerate(self.flat_boxes):
                    if self.stabilities[0, i, 0] > 0.5:
                        glVertex2f(box.body.position[0], box.body.position[1])
            elif not self.dropped_object == None:
                if self.stabilities[0,0,0] > 0.5:
                    glVertex2f(self.dropped_object.body.position[0], self.dropped_object.body.position[1])
                for (i, box) in enumerate(self.flat_boxes):
                    if self.stabilities[0, i+1, 0] > 0.5:
                        glVertex2f(box.body.position[0], box.body.position[1])
                
        glEnd()
            
    # This method returns true if two box is touching each other and false otherwise
    def there_is_relation(self, box_a, box_b):
        return (math.sqrt((box_a.body.position[0] - box_b.body.position[0])**2 + (box_a.body.position[1] - box_b.body.position[1])**2) < self.relation_threshold)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.event_loop.exit()

        elif symbol == key.SPACE:
            self.create_world()

        elif symbol == key.DOWN:
            self.drop_object() # Dropping an object from the highest part of the tower

        elif symbol == key.P:
            stabilities = self.calculate_stability(-1)
            print('stabilities calculated: {}'.format(stabilities))

        elif symbol == key.D and self.demolish:
            self.drop_to_demolish() # drop an object to the top of the tower to demolish the tower

        elif symbol == key.J and self.jenga:
            self.remove_object() # removes one random object from the system
        
        elif symbol == key.K and self.jenga and self.demolish:
            self.remove_to_demolish() # removes the object that is expected to have the least stability

        elif symbol == key.S:
            self.save_trajectories()
        

# This script runs the model and saves the trajectories if wanted
# Supposed to run in Python2
if __name__ == '__main__':
    towerCreator = TowerCreator(n=7, N=200, self_run=True)
    towerCreator.run()