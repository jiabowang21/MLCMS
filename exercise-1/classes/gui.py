import sys
import tkinter
import json
import copy
import time
from tkinter import ttk
import numpy as np
from tkinter import *
from tkinter import filedialog as FileDialog
import os
from .scenario import Scenario
from .pedestrian import Pedestrian
import csv


import random

class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the 'start_gui' method.
    """

    def update_reset_status(self, active):
        """
        Updates the reset status depending on the boolean value 'active'.
        If 'active', a copy of the current scenario is saved as a backup scenario and the state of the reset button changes its 
        state to NORMAL, which means that it can be clicked. Otherwise, the backup scenario is removed and the state changes to
        DISABLED, so it cannot be clicked.

        Parameters:
            active (bool): The reset status to be changed to.
        """
        if active:
            self.backup_scenario = self.scenario.copy()
            self.reset_button["state"] = NORMAL
        else:
            self.backup_scenario = None
            self.reset_button["state"] = DISABLED

    def create_scenario(self, width, height, root):
        """
        Creates a scenario with the dimensions specified by the user.
        The backup scenario for the reset is cleaned up.

        Parameters:
            width: Width of the new scenario
            heigth: Heigth of the new scenario
            root: Tkinter windows to be closed
        """
        try:
            width = int(width)
            height = int(height)
        except:
            print("ERROR: Dimensions must be integer values") 
            return
        
        if width < 1 or width > 1024 or height < 1 or height > 1024:
            print('ERROR: Please enter valid dimensions. Width and height must be in [1, 1024]')
            return

        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(width, height)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        
        self.scenario.to_image(self.canvas, self.canvas_image)
        self.update_reset_status(False)
        root.destroy()

    def create_scenario_gui(self):
        """
        Creates and shows a user interface which allows the user to create a new scenario by specifying its dimensions.
        """
        root = Tk()

        root.title('Create a New Scenario')
        root.geometry('')
        root.resizable(False, False)

        label = Label(root, text="Width")
        label.grid(row=0,column=0,sticky=W, padx=5, pady=5)

        entry = Entry(root)
        entry.grid(row=0,column=1, padx=5, pady=5)

        label2 = Label(root, text="Length")
        label2.grid(row=1,column=0,sticky=W, padx=5)

        entry2 = Entry(root)
        entry2.grid(row=1,column=1, padx=5)

        button_frame = Frame(root)
        button_frame.grid(row=2, column=1)

        button = Button(button_frame, text='Accept', command=lambda: self.create_scenario(entry.get(), entry2.get(), root))
        button.grid(row=0,column=1, pady=5)

        button = Button(button_frame, text='Cancel', command=root.destroy)
        button.grid(row=0,column=0, pady=5)

    def run(self, root):
        """
        Executes single steps recursively until no more steps are made.

        Parameters:
            root: The main window where the simulation runs. This is needed for updating the canvas and making a recursive call.
        """

        if self.running:
            updated = self.scenario.update_step()
            if not updated:
                self.running = False
                if len(self.scenario.pedestrian_records) > 1:
                    f = open('./out/pedestrian_records_extras.csv', 'w')
                    writer = csv.writer(f)
                    for row in self.scenario.pedestrian_records:
                        writer.writerow(row)
                    f.close()
                for k,v in self.scenario.measuring_records.items():
                    f = open('./out/'+str(k)+'_records.csv', 'w')
                    writer = csv.writer(f)
                    writer.writerow(v)
                    f.close()
            else:
                self.scenario.to_image(self.canvas, self.canvas_image)
                root.update()
            root.after(0, lambda: self.run(root))

    def run_simulation(self, root):
        """
        Saves a copy of the current scenario if there is no backup. The attribute `running`
        changes to `True` in order to execute recursively the `run()` function.
        """
        if self.backup_scenario == None:
            self.update_reset_status(True)
        self.running = True
        
        root.after(0, lambda: self.run(root))

    def step_scenario(self):
        """
        Saves a copy of the current scenario if there is no backup. Moves the simulation
        forward by one step and visualizes the result.
        """
        if self.backup_scenario == None:
            self.update_reset_status(True)
        self.scenario.update_step()
        self.scenario.to_image(self.canvas, self.canvas_image)

    def clear_scenario(self):
        """
        Clears all the elements of the scenario as well as the backup scenario.
        """
        self.scenario.pedestrians.clear()
        self.scenario.grid = np.zeros((self.scenario.width, self.scenario.height))

        self.update_reset_status(False)

        self.scenario.to_image(self.canvas, self.canvas_image)

    def reset_scenario(self):
        """
        Resets the scenario back to the one defined just before running a simulation or making the first step (backup scenario).
        The backup scenario is removed and the reset button changes its state to disabled.
        """
        self.running = False
        if self.backup_scenario != None:
            self.scenario = self.backup_scenario.copy()
            self.scenario.to_image(self.canvas, self.canvas_image)
            self.update_reset_status(False)

    def load_scenario(self):
        """
        Opens a file selector which allows to select a .json that defines a scenario and loads it.
        The restart status changes to inactive (backup scenario cleared and reset button disabled).
        The old scenario is overwrited with the new dimensions and elements defined in the .json file.
        """
        json_file = FileDialog.askopenfilename()
        file_name, file_extension = os.path.splitext(json_file)
        if file_name == '':
            return
        elif file_extension != '.json':
            print("ERROR: Please, select a .json file")
            return
        json_file = open(json_file)
        data = json.load(json_file)

        width = data["width"]
        height = data["height"]

        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(width, height)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.scenario.to_image(self.canvas, self.canvas_image)

        self.update_reset_status(False)

        pedestrians = data["pedestrians"]
        targets = data["targets"]
        obstacles = data["obstacles"]

        for pedestrian in pedestrians:
            x = pedestrian[0][0]
            y = pedestrian[0][1]
            speed = pedestrian[1]
            self.scenario.pedestrians.append(Pedestrian((x,y), speed))
            self.scenario.grid[x, y] = Scenario.NAME2ID['PEDESTRIAN']

        for target in targets:
            self.scenario.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']

        for obstacle in obstacles:
            self.scenario.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']

        self.scenario.recompute_target_distances()

        self.scenario.to_image(self.canvas, self.canvas_image)

    def add_pedestrian(self, position, desired_speed):
        """
        Adds a pedestrian to the scenario with the specified position and desired_speed

        Parameters:
            position: Position of the pedestrian
            desired_speed: Desired speed of the pedestrian
        """
        try:
            coordinate_x = int(position[0])
            coordinate_y = int(position[1])
            desiredSpeed = float(desired_speed)
        except:
            print("ERROR:") 
            print("  Coordinates and Desired Speed must be integer and float values respectively") 
            return

        if (coordinate_x < 0 or coordinate_x >= self.scenario.width) or (coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:") 
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  and")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        self.scenario.pedestrians.append(Pedestrian((coordinate_x, coordinate_y), desiredSpeed))
        self.scenario.grid[(coordinate_x, coordinate_y)] = Scenario.NAME2ID['PEDESTRIAN']
        self.scenario.to_image(self.canvas, self.canvas_image)

        self.update_reset_status(False)

    def remove_pedestrian(self, position):
        """
        Removes a pedestrian in the specified position from the scenario

        Parameters:
            position: Position of the pedestrian to remove
        """

        try:
            coordinate_x = int(position[0])
            coordinate_y = int(position[1])
        except:
            print("ERROR:") 
            print("  Coordinates must be integer values") 
            return

        if (coordinate_x < 0 or coordinate_x >=self.scenario.width) or (coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:")
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        for i in self.scenario.pedestrians:
            if (coordinate_x, coordinate_y) == i.position:
                self.scenario.pedestrians.remove(i)
                self.scenario.grid[(coordinate_x, coordinate_y)] = Scenario.NAME2ID['EMPTY']

        self.scenario.to_image(self.canvas, self.canvas_image)
        self.update_reset_status(False)

    def edit_pedestrians_gui(self, root):
        """
        Places the widgets required for editing pedestrians to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """
        
        label = Label(root, text="Coordinate X")
        label.grid(row=1,column=0, sticky=W, padx=5)

        entry = Entry(root)
        entry.grid(row=1,column=1)

        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=2,column=0, sticky=W, padx=5)

        entry2 = Entry(root)
        entry2.grid(row=2,column=1)

        label3 = Label(root, text="Desired Speed")
        label3.grid(row=3,column=0, sticky=W, padx=5)

        entry3 = Entry(root)
        entry3.grid(row=3,column=1)

        button_frame = Frame(root)
        button_frame.grid(row=4, column=1)

        button = Button(button_frame, text='Add', command=lambda: self.add_pedestrian((entry.get(), entry2.get()), entry3.get()))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=1)

        button = Button(button_frame, text='Remove', fg='red',command=lambda: self.remove_pedestrian((entry.get(), entry2.get())))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=0)

    def edit_target_or_obstacle(self, position_x, position_y,thing):
        """
        Adds or removes a target or an obstacle from the specified position, depending on the parameter `thing`

        Parameters:
            position_x: x position
            position_y: y position
            thing: the element to place
        """
        try:
            coordinate_x = int(position_x)
            coordinate_y = int(position_y)
        except:
            print("ERROR:") 
            print("  Coordinates must be integer values") 
            return


        if (coordinate_x < 0 or coordinate_x >= self.scenario.width) or (coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:") 
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  and")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        self.scenario.grid[coordinate_x, coordinate_y] = Scenario.NAME2ID[thing]
        self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)

        self.update_reset_status(False)

    def edit_targets_gui(self, root):
        """
        Places the widgets required for editing targets to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """
        label = Label(root, text="Coordinate X")
        label.grid(row=6,column=0, sticky=W, padx=5)

        entry = Entry(root)
        entry.grid(row=6,column=1)
        
        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=7,column=0, sticky=W, padx=5)

        entry2 = Entry(root)
        entry2.grid(row=7,column=1)
        
        button_frame = Frame(root)
        button_frame.grid(row=8, column=1)

        button = Button(button_frame, text='Add', command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'TARGET'))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=1)

        button = Button(button_frame, fg='red', text='Remove', command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'EMPTY'))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=0)

    def edit_obstacles_gui(self, root):
        """
        Places the widgets required for editing obstacles to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """
        label = Label(root, text="Coordinate X")
        label.grid(row=10,column=0, sticky=W, padx=5)

        entry = Entry(root)
        entry.grid(row=10,column=1)
        
        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=11,column=0, sticky=W, padx=5)

        entry2 = Entry(root)
        entry2.grid(row=11,column=1)

        button_frame = Frame(root)
        button_frame.grid(row=12, column=1)
        
        button = Button(button_frame, text='Add', command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'OBSTACLE'))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=1)

        button = Button(button_frame, fg='red',text='Remove', command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'EMPTY'))
        button.config( height = 1, width = 5 )
        button.grid(row=0, column=0)

    def edit_scenario_gui(self):
        """
        Creates and shows a user interface which allows the user to edit the elements of the scenario.
        Supported functions are adding or removing pedestrians, targets and obstacles.
        """

        root = Tk()
        root.resizable(False, False) 

        root.title('Edit Elements of the Scenario')
        root.geometry('')

        label = Label(root, text="Pedestrian")
        label.grid(row=0,column=0,sticky=W, padx=5, pady=5)

        self.edit_pedestrians_gui(root)

        label2 = Label(root, text="Target")
        label2.grid(row=5,column=0,sticky=W, padx=5, pady=5)     

        self.edit_targets_gui(root)

        label3 = Label(root, text="Obstacle")
        label3.grid(row=9,column=0,sticky=W, padx=5, pady=5)     

        self.edit_obstacles_gui(root)

        button = Button(root, text="Done", command=root.destroy)
        button.grid(row=13,column=2, padx=5, pady=5)

        root.mainloop()

    def fill_pedestrians(self, start_position, size, pedestrians_num, pedestrians_speed):
        """
        Fill a given area in the scenario with a given number of pedestrians uniformly and randomly distributed.
        
        Parameters:
            start_position: A tuple of starting point of the area to be filled
            size:           A tuple of the size of the area to be filled
            pedestrians_num: number of the filled pedestrians
            pedestrians_speed: speed of the filled pedestrians
        """
        pedestrians = []
        for pos in random.sample(range(size[0]*size[1]), pedestrians_num):
            pedestrians.append(Pedestrian((start_position[0]+pos%size[0], start_position[1]+pos//size[0]), pedestrians_speed))
        return pedestrians

    def task_2(self):
        """
        Loads a scenario related to the task 2. Restart status changes to inactive.
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(50, 50)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        self.scenario.grid[25, 25] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        self.scenario.pedestrians = [
            Pedestrian((5, 25), 1)
        ]

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)

    def task_3(self):
        """
        Loads a scenario related to task 3. Restart status changes to inactive.
        """

        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(50, 50)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        self.scenario.grid[24, 24] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        self.scenario.pedestrians = [
            Pedestrian((0, 6), 1),
            Pedestrian((0, 42), 1),
            Pedestrian((6, 0), 1),
            Pedestrian((48, 6), 1),
            Pedestrian((48, 42), 1)
        ]

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)

    def task_4(self):
        """
        Loads a scenario related to task 4. Restart status changes to inactive.
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(100, 100)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)


        for i in range(40):
            self.scenario.grid[i, 0] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[60 + i, 0] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[i, 40] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[60 + i, 40] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[0, i] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[99, i] = Scenario.NAME2ID['OBSTACLE'] 
            
            self.scenario.grid[30 + i, 50] = Scenario.NAME2ID['OBSTACLE']
            self.scenario.grid[30 + i, 90] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[69, 50 + i] = Scenario.NAME2ID['OBSTACLE']


        for i in range(20):
            self.scenario.grid[40 + i, 17] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[40 + i, 23] = Scenario.NAME2ID['OBSTACLE'] 
        
        for i in range(18):
            self.scenario.grid[40, i] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[40, i + 23] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[60, i] = Scenario.NAME2ID['OBSTACLE'] 
            self.scenario.grid[60, i + 23] = Scenario.NAME2ID['OBSTACLE']

        for i in range(6):
            self.scenario.grid[99, 17 + i] = Scenario.NAME2ID['TARGET']
            self.scenario.grid[95, 67 + i] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        self.scenario.pedestrians = []
        self.scenario.pedestrians += self.fill_pedestrians((1, 1), (20, 39), 150, 1)
        self.scenario.pedestrians += self.fill_pedestrians((10, 60), (40, 20), 150, 1)

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)

    def scenario_1(self):
        """
        Loads the scenario 1 of the RiMEA guidelines.

        Description:
            It is to be proven that a person in a 2m wide and 40 m long corridor with a
            defined walking speed will cover the distance in the corresponding time period.
            Scenario is 100 x 100 cells, each cell is 40 cm, so it is 40m x 40m 
            The corridor is a 5 cells height (2m) x 100 cells wide (40m) space in the middle
            of the scenario
            The only pedestrian is located at the beggining of the corridor (0, 50)
            with speed 3.25 cells/step
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(100, 100)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        # obstacle positions
        for i in range(0,100):
            for j in range(0, 48):
                self.scenario.grid[i,j] = Scenario.NAME2ID['OBSTACLE'] 

            for j in range(53, 100):
                self.scenario.grid[i,j] = Scenario.NAME2ID['OBSTACLE'] 

        # target positions
        for i in range(48, 53):
            self.scenario.grid[99,i] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        # pedestrian position
        self.scenario.pedestrians = [Pedestrian((0, 50), 3.25)]

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)
 
                
    def scenario_4(self):
        """
        Loads the scenario 4 of the RiMEA guidelines.
        
        (Assuming each 3*3 pixel is representing 1 m^2)
        Parameter:
            density: the density of pedestrians to be simulated.

        Description:
            A corridor (1000 m long, 10 m wide) is to be filled with different densities 
            of persons with an equal as possible free walking 
            speed (for example 1.2 – 1.4m/s): 0.5 P/m², 1 P/m², 
            2 P/m², 3 P/m², 4 P/m², 5 P/m² and 6 P/m²        
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(300, 300)
        w = 300
        h = 300

        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        for i in range(300):
            self.scenario.grid[i, 134] = Scenario.NAME2ID['OBSTACLE']
            self.scenario.grid[i, 165] = Scenario.NAME2ID['OBSTACLE']

        for i in range(30):
            self.scenario.grid[w-2,135+i] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        self.scenario.pedestrians = []
        for i in range(30):
            for j in range(5):
                # this is to make sure of filling an integer number of pedestrians into 6*6 (2m*2m) grids, for the density might be 0.5 P/m².
                pedestrians_num = int(self.density * 4)
                self.scenario.pedestrians += self.fill_pedestrians((i*6, j*6+135), (6, 6), pedestrians_num, 4)

        self.scenario.measuring_points = [(120,147,6),(270,147,6),(270,153,6)]
        for measuring_point in self.scenario.measuring_points:
            self.scenario.measuring_records[measuring_point] = []

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)

    def scenario_6(self):
        """
        Loads the scenario 6 of the RiMEA guidelines.

        Description:
            Twenty persons moving towards a corner which turns to the left 
            that successfully go around it without passing through walls.
            The scenario is 24 x 24 cells, each cell is 0.5 meter, so the scenario is 12m x 12m 
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        # scenario size of 24x24
        self.scenario = Scenario(24, 24)

        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        # targets positions 
        for i in range(20, 24):
            self.scenario.grid[i, 0] = Scenario.NAME2ID['TARGET']

        # obstacle positions
        for i in range(0,20):
            for j in range(0, 20):
                self.scenario.grid[i,j] = Scenario.NAME2ID['OBSTACLE'] 

        self.scenario.recompute_target_distances()
        
        # pedestrians positions (20 pedestrians)
        for i in range(0, 5):
            for j in range(20, 24):
                self.scenario.pedestrians.append(Pedestrian((i, j), 1))

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)
        
    def scenario_7(self):
        """
        Loads the scenario 7 of the RiMEA guidelines.
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(50, 50)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method

        self.update_reset_status(False)

        for i in range(0, 50):
            self.scenario.grid[49, i] = Scenario.NAME2ID['TARGET']
            self.scenario.pedestrians.append(Pedestrian((0, i), 1))
            self.scenario.pedestrians[i].id = i

        self.scenario.assign_ages()
        self.scenario.assign_speeds()

        self.scenario.recompute_target_distances()

        self.scenario.to_image(self.canvas, self.canvas_image)

    def change_algorithm(self, *args):
        """
        Changes the algorithm used to compute the target distances depending on the option selected in the option menu.
        The options might be 'Basic', 'Dijkstra' or 'FMM'. 
        Parameters:
            args: contains the option displayed in the option menu
        """
        self.scenario.recompute_method = args[0].upper()
        self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)

        if self.backup_scenario != None:
            self.backup_scenario.recompute_method = args[0].upper()
            self.backup_scenario.recompute_target_distances()

    def change_view(self, value):
        """
        Changes the view of the scenario depending on the value of the check button.
        If value = 1 then the view of the scenario changes to the target distance-based view, otherwise changes to normal view.
        Parameters:
            value (int): the value of the check button.
        """
        if value == 1:
            self.scenario.target_grid = True
        else:
            self.scenario.target_grid = False
        if self.backup_scenario != None:
            self.backup_scenario.target_grid = self.scenario.target_grid
            
        self.scenario.to_image(self.canvas, self.canvas_image)

    def change_density(self, *args):
        self.density = float(args[0])

    def start_gui(self):
        """
        Creates and shows a user interface with multiple buttons and a canvas where
        the scenario of the Celullar Automata is displayed.
        The scenario is initialized with dimensions width = length = 100 and no elements.
        The buttons implemented are the following:
        - New: creates a new scenario with the specified dimensions
        - Load: allows the user to load a .json file that defines a scenario
        - Edit: allows the user to add/remove elements (pedestrians, targets or obstacles) from the scenario
        - Clear: removes all the elements of the scenario
        - Change algorithm: allows the user to change the algorithm used for the computation of distances
        - Change view: allows the user to alternate between the normal and target grid distance-based views
        - Run: runs a simulation of the defined scenario
        - Step: makes a single step of the simulation of the defined scenario
        - Reset: allows the user to restart the scenario back to the one defined just before running a simulation or making the first step
        - Task 2/3/4: loads the pre-defined scenario related to the corresponding task
        - Scenario 1/4/6/7: loads the corresponding pre-defined scenario according to the RiMEA guidelines.
        """
        # Initialize Tkinter window
        win = tkinter.Tk()
        win.geometry("")
        win.resizable(False, False)
        win.title('Cellular Automata GUI')

        # Initialize scenario
        self.scenario = Scenario(100, 100)
        self.backup_scenario = None
        self.running = False
        
        # Set up frames for buttons and canvas 
        button_frame = Frame(win)
        button_frame.pack(side='left', expand=True, fill=BOTH)
        canvas_frame = Frame(win)
        canvas_frame.pack(side='left', expand=True, fill=BOTH)

        # Buttons-related functions
        button = Button(button_frame, text='New', command=lambda: self.create_scenario_gui())
        button.config( height = 1, width =10 )
        button.grid(row=0, column=0)

        button = Button(button_frame, text='Load', command=lambda: self.load_scenario())
        button.config( height = 1, width =10 )
        button.grid(row=1, column=0)

        button = Button(button_frame, text='Edit', command=lambda: self.edit_scenario_gui())
        button.config( height = 1, width =10 )
        button.grid(row=2, column=0)

        button = Button(button_frame, text='Clear', command=lambda: self.clear_scenario())
        button.config( height = 1, width =10 )
        button.grid(row=3, column=0)

        label = Label(button_frame, text="")
        label.grid(row=4, column=0)

        algorithms = ['FMM', 'Dijkstra', 'Basic']
        selected_algorithm = tkinter.StringVar(button_frame)
        selected_algorithm.set(algorithms[0])
        option_menu = OptionMenu(button_frame, selected_algorithm, *algorithms, command=self.change_algorithm)
        option_menu.config(height=1, width=5)
        option_menu.grid(row=5, column=0)

        check_value = tkinter.IntVar()
        check_button = Checkbutton(button_frame, text='Show target grid', variable = check_value, command=lambda: self.change_view(check_value.get()))
        check_button.grid(row=6, column =0)

        label = Label(button_frame, text='')
        label.grid(row=7, column=0)

        button = Button(button_frame, text='Run', command=lambda: self.run_simulation(win))
        button.config( height = 1, width =10 )
        button.grid(row=8, column=0)

        button = Button(button_frame, text='Step', command=lambda: self.step_scenario())
        button.config( height = 1, width =10 )
        button.grid(row=9, column=0)

        self.reset_button = Button(button_frame, text='Reset', command=lambda: self.reset_scenario())
        self.reset_button.config( height = 1, width =10 )
        self.reset_button.grid(row=10, column=0)
        self.reset_button["state"] = DISABLED
        
        label = Label(button_frame, text='')
        label.grid(row=11, column=0)

        button = Button(button_frame, text='Task 2', command=lambda: self.task_2())
        button.config( height = 1, width =10 )
        button.grid(row=12, column=0)

        button = Button(button_frame, text='Task 3', command=lambda: self.task_3())
        button.config( height = 1, width =10 )
        button.grid(row=13, column=0)

        button = Button(button_frame, text='Task 4', command=lambda: self.task_4())
        button.config( height = 1, width =10 )
        button.grid(row=14, column=0)

        label = Label(button_frame, text='')
        label.grid(row=15, column=0)

        button = Button(button_frame, text='Scenario 1', command=lambda: self.scenario_1())
        button.config( height = 1, width =10 )
        button.grid(row=16, column=0)

        button = Button(button_frame, text='Scenario 4', command=lambda: self.scenario_4())
        button.config( height = 1, width =10 )
        button.grid(row=17, column=0)

        scenario_4_frame = Frame(button_frame)
        scenario_4_frame.grid(row=18, column=0)

        label = Label(scenario_4_frame, text='Density')
        label.pack(side = 'left')

        self.density = 0.5
        densities = ['0.5', '1', '2', '3', '4', '5', '6']
        selected_density = tkinter.StringVar(button_frame)
        selected_density.set(densities[0])
        option_menu = OptionMenu(scenario_4_frame, selected_density, *densities, command=self.change_density)
        option_menu.config(height=1, width=1)
        option_menu.pack(side='left')

        button = Button(button_frame, text='Scenario 6', command=lambda: self.scenario_6())
        button.config( height = 1, width =10 )
        button.grid(row=19, column=0)

        button = Button(button_frame, text='Scenario 7', command=lambda: self.scenario_7())
        button.config( height = 1, width =10 )
        button.grid(row=20, column=0)

        label = Label(button_frame, text='')
        label.grid(row=21, column=0)

        button = Button(button_frame, text='Exit', command=self.exit_gui)
        button.config( height = 1, width =10 )
        button.grid(row=22, column=0)

        # Canvas-related functions
        canvas_width, canvas_height = Scenario.GRID_SIZE[0]+5, Scenario.GRID_SIZE[1]+5
        self.canvas = Canvas(canvas_frame, bd=0, width=canvas_width, height=canvas_height)  # creating the canvas
        self.canvas_image = self.canvas.create_image(canvas_width/2,canvas_height/2, image=None, anchor=tkinter.CENTER)
        self.canvas.pack(side=LEFT,expand=True,fill=BOTH)

        self.scenario.to_image(self.canvas, self.canvas_image)
        
        win.mainloop()
    
    def exit_gui(self):
        """
        Closes the GUI.
        """
        sys.exit()
