# This model is the same as the first model. 
# Only it has a graph neural network running at the background and it does some predictions.
# A graph is put into the objects with each object representing a node in the graph. 
# A line is drawn between the centers of the objects if they have a relationship.
# The objects that are predicted to stay still is colored in yellow and the others are colored in blue. TODO: look into this comment
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

class TowerCreator(pyglet.window.Window):
    def __init__(self, n, N, self_run=False, predict_stability=False, gnn_model=None):
        self.n = n # Number of rectangles
        self.N = N # Number of iterations
        self.self_run = self_run
        self.predict_stability = predict_stability
        self.predicted_stability = False
        self.gnn_model = gnn_model
        self.stabilities = []

        self.window_width = 800
        self.window_height = 600

        self.event_loop = pyglet.app.EventLoop()
        pyglet.window.Window.__init__(self, self.window_width,
                                        self.window_height, vsync=False)

        self.set_caption('Random tower building')

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
        print(self.relation_threshold)

        self.left_edge = 100 # the left edge of the x point i want to put rectangles to
        self.right_edge = 700 # The space for putting the rectangles is from x (100,700)
        self.spacing = self.right_edge - self.left_edge 
        self.bottom_edge = 70

        # X positions of the rectangles are chosen with not a random position but
        # with a random interval to put the box
        # And when a random interval is chosen by a box option for that place
        # is removed
        # ex: if the left/right edges and the interval is as above (100,700 and width is 100)
        # then the rects can be at x: 100, 200, 300,400,500, 600 or 700.
        self.random_x_interval = int(self.spacing / self.rect_width)

        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.draw_options.collision_point_color = (10, 20, 30, 40)

        # trajectories will look like: 
        # dropped_object: {X: [...], Y: [...]}, boxes: [ {X: [...], Y: [...]}, {X: [...], Y: [...]}, ... ]
        # self.trajectories = {'dropped_object': {'X':[], 'Y':[]}}
        # self.trajectories['boxes'] = [{'X':[], 'Y':[]} for i in range(self.n)]
        # print('self.trajectories is: {}'.format(self.trajectories))
        # trajectories will look like: 
        # [trajectory#N):[(object#1):[(X):[...], (Y):[...]], (object#n):[(X):[], (Y):[]]...], trajectory#N+1:[.........], ...]
        self.trajectories = []

        if self.self_run:
            self.run_and_take_trajectory()
        else:
            self.create_world() # Creates a world with randomly put n rectangles

        # self.create_world()

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        self.event_loop.run()

    def run_and_take_trajectory(self):
        for i in range(self.N):
            pyglet.clock.schedule_once(self.callback, i*4, callback_type=0)
            pyglet.clock.schedule_once(self.callback, i*4+1, callback_type=1)
        
        pyglet.clock.schedule_once(self.callback, self.N*4, callback_type=2)
        pyglet.clock.schedule_once(self.event_loop.exit, self.N*4+3)
        print('*** scheduling over')

    def callback(self, dt, callback_type):
        if callback_type == 0:
            self.create_world()
        elif callback_type == 1:
            self.drop_object()
        elif callback_type == 2:
            self.save_trajectories()

    def save_trajectories(self):
        # create random endix to the file name
        letters_and_digits = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(letters_and_digits) for i in range(8))
        # dump the trajectories into a file
        file_name = 'data/first_model_{}_{}_{}.txt'.format(self.n, self.N, random_string)
        with open(file_name, 'w+') as outfile:
            json.dump(self.trajectories, outfile)

    def create_world(self):
        print('in create_world')
        # print('len(trajectories): {}'.format(len(self.trajectories)))
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0., -900.)
        self.space.sleep_time_threshold = 0.9
        self.boxes = [] # 2d array box instances correspondingly for each layer
        self.dropped_object = None
        self.dropped_trajectory = [[],[]] # First element represents x trajectory, second: y
        self.predicted_stability = False

        # Create the ground line
        ground_line = pymunk.Segment(self.space.static_body, Vec2d(20, self.bottom_edge), Vec2d(780, self.bottom_edge), 1)
        ground_line.friction = 0.9
        self.space.add(ground_line)

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
                fitter = 3
                pre_r = random.randint(1, fitter * min(layers[j-1]+1, n))
                if pre_r <= fitter and pre_r != 1:
                    r = random.randint(min(layers[j-1]+1, n, 2), min(layers[j-1]+1, n))
                elif pre_r > fitter:
                    r = math.floor(pre_r / fitter)
                else:
                    r = pre_r # pre_r == 1
            layers.append(r)
            n -= r
            j += 1
        print('number of boxes in each layer is: {}'.format(layers))

        for (layer_num, layer_size) in enumerate(layers):
            self.boxes.append([])
            middle_x = self.get_middle(layer_num)
            self.put_boxes(layer_num, layer_size, middle_x)

        self.flat_boxes = []
        for i in range(len(self.boxes)):
            for j in range(len(self.boxes[i])):
                self.flat_boxes.append(self.boxes[i][j])
        print('len(flat_boxes): {}'.format(len(self.flat_boxes)))
        print('len(trajectories): {}'.format(len(self.trajectories)))
        if len(self.flat_boxes) == self.n:
            self.trajectories.append([])

    # This method puts necessary number of boxes to the given layer by starting putting the box in the middle to the middle position
    def put_boxes(self, layer_num, layer_size, middle_x):
        for i in range(layer_size):
            box_variation = int(self.rect_width * 0.20)
            mean_range = self.rect_width + 2 * box_variation
            box_mean = middle_x + ( ((-1) ** i) * math.floor((i+1)/2) * mean_range ) 
            # print('box_variation: {}, box_mean: {}'.format(box_variation, box_mean))
            if layer_size % 2 == 0:
                x_pos = random.randint(box_mean - box_variation, box_mean + box_variation) + int(mean_range / 2)
            else:
                x_pos = random.randint(box_mean - box_variation, box_mean + box_variation)
            y_pos = self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num

            mass = 50.0
            moment = pymunk.moment_for_box(mass, (self.rect_width, self.rect_height))
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
        right_edge = -10000 # The window width cannot be 10000 pixels for sure
        left_edge = 10000
        for box in self.boxes[layer_num-1]:
            if box.body.position[0] > right_edge: 
                right_edge = box.body.position[0]
            if box.body.position[0] < left_edge:
                left_edge = box.body.position[0]
        return int((left_edge + right_edge) / 2)

    # Drop a random object to the tower, x position is randomly found
    def drop_object(self):
        print('in drop_object')
        x_pos = random.randint(self.left_edge, self.right_edge)
        layer_num = len(self.boxes)
        y_pos = self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num
        mass = 50.0
        moment = pymunk.moment_for_box(mass, (self.rect_width, self.rect_height))
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(x_pos, y_pos)
        shape = pymunk.Poly.create_box(body, (self.rect_width, self.rect_height))
        shape.friction = 0.3
        self.space.add(body, shape)
        self.dropped_object = shape

    # This function returns an array (n_objects), indicating the stability of each object
    # This is called at the of the drop_object
    def predict_stabilities(self):
        n_of_traj = 1
        n_objects = self.n+1 # including the dropped object
        n_relations = n_objects * (n_objects - 1)
        n_object_attr_dim = 2

        boxes = np.zeros((1, n_objects, n_object_attr_dim)) # 1 for 1 trajectory
        for o in range(n_objects):
            t = len(self.trajectories)-1
            print('len(self.trajectories[0]): {}, o: {}'.format(len(self.trajectories[0]), o))
            boxes[0,o,0] = self.trajectories[t][o][0][0] / 170.0
            boxes[0,o,1] = self.trajectories[t][o][0][1] / 170.0

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
                    inzz = np.linalg.norm(boxes[:,m,0:2] - boxes[:,j,0:2], axis=1) < relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt] = 1.0   
                    cnt += 1

        self.stabilities = self.gnn_model.predict({'objects': boxes[0:1,:,:], 'sender_relations': val_sender_relations,
                                                    'receiver_relations': val_receiver_relations, 'propagation': propagation})

    def update(self, dt):
        step_dt = 1/250.
        x = 0
        while x < dt:
            x += step_dt
            self.space.step(step_dt)

        if len(self.flat_boxes) == self.n:
            if not self.dropped_object == None:
                if len(self.trajectories[-1]) == 0:
                    self.trajectories[-1].append([])
                    for _ in range(self.n):
                        self.trajectories[-1].append([])
                    
                self.trajectories[-1][0].append([self.dropped_object.body.position[0],self.dropped_object.body.position[1]])
                for i,box in enumerate(self.flat_boxes):
                    self.trajectories[-1][i+1].append([box.body.position[0],box.body.position[1]])

            if self.predict_stability and not self.predicted_stability:
                if not self.dropped_object == None:
                    self.predict_stabilities()
                    print('self.stabilities: {}'.format(self.stabilities))
                    self.predicted_stability = True

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
            self.draw_drop_object_trajectory()

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
        
        glPointSize(50)
        glBegin(GL_POINTS)
        glColor3i(255, 0, 0)
        # if not len(self.stabilities) == 0:
        #     if self.stabilities[0, i, 0] > 0.5:
        #         box_a.color = (255,0,0)
        if self.predicted_stability:
            if self.stabilities[0,0,0] > 0.5:
                glVertex2f(self.dropped_object.body.position[0], self.dropped_object.body.position[1])
            for (i, box) in enumerate(self.flat_boxes):
                if self.stabilities[0, i+1, 0] > 0.5:
                    glVertex2f(box.body.position[0], box.body.position[1])
            
        glEnd()
            

# This script runs the model and saves the trajectories if wanted
# Supposed to run in Python2
if __name__ == '__main__':
    n = int(input('Please enter the number of rectangles you want: '))
    N = 1
    self_run_str = raw_input('Would you like to interract with the app or let the app run itself and have trajectory? [y/n]')
    print(self_run_str)

    if self_run_str == 'y':
        N = int(input('Please enter the number of iterations you want for this n: '))
        self_run = True
    else:
        self_run = False

    towerCreator = TowerCreator(n, N, self_run)
    towerCreator.run()