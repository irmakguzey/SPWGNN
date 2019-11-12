# This model creates n number of random rectangles with a fixed size but in random positions. 
# And it creates this environment N times. 
# And in each iteration it drops one more rectangle and looks saves the rectangles'
# positions and orientations (trajectories). 
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

class Main(pyglet.window.Window):
	def __init__(self, self_run, n, N):
		self.n = n # Number of rectangles
		self.N = N # Number of iterations
		self.self_run = self_run
		self.window_width = 800
		self.window_height = 600

		self.event_loop = pyglet.app.EventLoop()
		pyglet.window.Window.__init__(self, self.window_width,
									   self.window_height, vsync=False)

		self.set_caption('Random tower building')

		self.fps_display = pyglet.clock.ClockDisplay()

		self.iteration_text = pyglet.text.Label('Number of iterations is: {}'.format(self.N),
							font_size=10,
							x=10, y=400)
		self.rectangle_text = pyglet.text.Label('Number of rectangles is: {}'.format(self.n),
							font_size=10,
							x=10, y=450)

		self.rect_width = 150 # set size for rectangles
		self.rect_height = 60

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

		# trajectories will look like: 
		# dropped_object: {X: [...], Y: [...]}, boxes: [ {X: [...], Y: [...]}, {X: [...], Y: [...]}, ... ]
		self.trajectories = {'dropped_object': {'X':[], 'Y':[]}}
		self.trajectories['boxes'] = [{'X':[], 'Y':[]} for i in range(self.n)]
		print('self.trajectories is: {}'.format(self.trajectories))

		if self.self_run:
			self.run_and_take_trajectory()
		else:
			self.create_world() # Creates a world with randomly put n rectangles

		# self.create_world()

	def run(self):
		pyglet.clock.schedule_interval(self.update, 1/60.0)
		self.event_loop.run()

	# TODO: create trajectory file
	def run_and_take_trajectory(self):
		for i in range(self.N):
			pyglet.clock.schedule_once(self.callback, i*4, callback_type=0)
			pyglet.clock.schedule_once(self.callback, i*4+1, callback_type=1)
		
		pyglet.clock.schedule_once(self.callback, self.N*4, callback_type=2)
		pyglet.clock.schedule_once(self.event_loop.exit, self.N*4+3)
		print('*** scheduling over')
		# new_event_loop.exit()

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
		with open(file_name, 'w') as outfile:
			json.dump(self.trajectories, outfile)

	def create_world(self):
		self.space = pymunk.Space()
		self.space.gravity = Vec2d(0., -900.)
		self.space.sleep_time_threshold = 0.9
		self.boxes = [] # 2d array box instances correspondingly for each layer
		self.dropped_object = None
		self.dropped_trajectory = [[],[]] # First element represents x trajectory, second: y


		# Create the ground line
		ground_line = pymunk.Segment(self.space.static_body, Vec2d(20, self.bottom_edge), Vec2d(780, self.bottom_edge), 1)
		ground_line.friction = 0.9
		self.space.add(ground_line)

		# There is a restriction in the program that the number of boxes in one
		# layer can never be higher than the number of boxes at the above layer
		# Setting up the number of boxes at the layers:
		layers = [random.randint(self.random_x_interval/2+1, self.random_x_interval)] # represents the number of boxes in each layer
		n = self.n - layers[0] # number of boxes in total
		j = 1
		while n > 0:
			r = random.randint(1, min(self.random_x_interval, n))
			while r > layers[j-1]:
				r = random.randint(1, min(self.random_x_interval, n))
			layers.append(r)
			n -= r
			j += 1
		print('number of boxes in each layer is: {}'.format(layers))

		for (layer_num, layer_size) in enumerate(layers):
			self.boxes.append([])
			for box_index in range(layer_size):
				x_pos = random.randint(self.left_edge, self.right_edge)
				try_number = 0
				try_exceed = 100
				if layer_num == 0: # It is more important to have a good ground
					try_exceed = 1000
				while try_number < try_exceed and (not self.lower_layers_check(x_pos, layer_num) or not self.same_layer_check(x_pos, layer_num)):
					# print('{}th box in {}th layer, {}th try'.format(box_index, layer_num, try_number))
					x_pos = random.randint(self.left_edge, self.right_edge)
					try_number += 1
				if try_number == try_exceed:
					# print('number of tries exceeded :)')
					continue # If after 50 random tries object wasn't able to be put then it doesnt put it :)
				y_pos = self.bottom_edge + self.rect_height/2 + self.rect_height * layer_num
				mass = 50.0
				moment = pymunk.moment_for_box(mass, (self.rect_width, self.rect_height))

				body = pymunk.Body(mass, moment)
				body.position = Vec2d(x_pos, y_pos)
				shape = pymunk.Poly.create_box(body, (self.rect_width, self.rect_height))
				shape.friction = 0.3
				self.space.add(body, shape)
				self.boxes[layer_num].append(shape)

		self.flat_boxes = []
		for i in range(len(self.boxes)):
			for j in range(len(self.boxes[i])):
				self.flat_boxes.append(self.boxes[i][j])

	# Drop a random object to the tower, x position is randomly found
	def drop_object(self):
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

	# This method checks whether a box is stable or not
	# If the layer is bigger than 0, then it checks whether there are two 
	# boxes under the box or whether 
	# This method only looks to the layers above the box not to once on the same level
	def lower_layers_check(self, x, layer):
		if layer == 0:
			return True
		lower_boxes = self.boxes[layer-1]
		# Check whether there is a box that under this one which can cary it by itself
		for first_box in lower_boxes:
			first_box_pos = first_box.body.position
			if abs(first_box_pos[0] - x) < self.rect_width/2: # means that box under can carry the box to come by itself
				return True
			if abs(first_box_pos[0] - x) < self.rect_width: # if this is the case box to come look for another box as well to carry it
				for second_box in lower_boxes:
					if second_box == first_box:
						continue
					second_box_pos = second_box.body.position
					if abs(second_box_pos[0] - x) < self.rect_width:
						return True
		return False

	# Check whether there is enough place for the box 
	def same_layer_check(self, x, layer):
		same_level_boxes = self.boxes[layer]
		for box in same_level_boxes:
			if abs(box.body.position[0] - x) < self.rect_width:
				return False
		return True

	def update(self, dt):
		step_dt = 1/250.
		x = 0
		while x < dt:
			x += step_dt
			self.space.step(step_dt)

		if len(self.flat_boxes) == self.n:
			for i,box in enumerate(self.flat_boxes):
				self.trajectories['boxes'][i]['X'].append(box.body.position[0])
				self.trajectories['boxes'][i]['Y'].append(box.body.position[1])
		if not self.dropped_object == None:
			self.trajectories['dropped_object']['X'].append(self.dropped_object.body.position[0])
			self.trajectories['dropped_object']['Y'].append(self.dropped_object.body.position[1])

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

if __name__ == '__main__':
	n = int(input('Please enter the number of rectangles you want: '))
	N = int(input('Please enter the number of iterations you want for this n: '))
	self_run_str = input('Would you like to interract with the app or let the app run itself and have trajectory? [y/n]')
	print(self_run_str)

	if self_run_str == 'y':
		self_run = True
	else:
		self_run = False

	main = Main(self_run, n, N)
	main.run()