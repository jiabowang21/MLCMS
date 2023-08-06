from .scenario import Scenario
import math
import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculates the euclidean distance of two points.

    Parameters:
        x1: x parameter of point 1
        y1: y parameter of point 1
        x2: x parameter of point 2
        y2: y parameter of point 2
    """
    x = x1-x2
    y = y1-y2
    return math.sqrt(x*x+y*y)


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self.accumulated_distance = 0
        self.id = -1
        self.age = -1
        self.distance_walked = 0
        self.ticks = 0
        self.finished = False
        self.saved = False

    def copy(self):
        return Pedestrian(self._position, self._desired_speed)

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.

        Parameters:
            scenario: The scenario instance.
        
        Return:
            A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        ]

    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.

        Parameters:
            scenario: The current scenario instance.
        """
        if self.finished:
            if not self.saved:
                self.saved = True
                if self.ticks == 0:
                    self.ticks = 1
                if self.id != -1:
                    scenario.pedestrian_records.append([self.id, self.age, self.desired_speed, self.distance_walked/self.ticks])
            return False
        
        moved = False
        available_distance = self._desired_speed + self.accumulated_distance
        self.accumulated_distance = 0
        while available_distance > 0:
            neighbors = self.get_neighbors(scenario)
            p_x = self._position[0]
            p_y = self._position[1]
            next_cell_distance = scenario.target_distance_grids[p_x, p_y]
            next_pos = self._position
            # Goes to the neighbor position that minimizes the distance to the nearest target
            for (n_x, n_y) in neighbors:
                if scenario.grid[n_x, n_y] != Scenario.NAME2ID['OBSTACLE'] and scenario.grid[n_x, n_y] != Scenario.NAME2ID['PEDESTRIAN']:
                    if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                        next_pos = (n_x, n_y)
                        next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                    
                    elif next_cell_distance == scenario.target_distance_grids[n_x, n_y]:
                        distance_to_next = euclidean_distance(p_x, p_y, next_pos[0], next_pos[1])
                        distance_to_neighbor = euclidean_distance(p_x, p_y, n_x, n_y)
                        if distance_to_neighbor < distance_to_next:
                            next_pos = (n_x, n_y)
                            next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                    
            if self._position != next_pos:
                '''
                    measuring_points related
                '''
                for x, y, size in scenario.measuring_points:
                    if  next_pos[0] >= x and next_pos[0] <= x + size and next_pos[1] >= y and next_pos[1] <= y + size and self._position[0] < x:
                        scenario.measuring_records[(x, y, size)] += [self.ticks]

                moved = True
                distance_to_travel = euclidean_distance(p_x, p_y, next_pos[0], next_pos[1])
                if distance_to_travel <= available_distance:
                    available_distance -= distance_to_travel
                    self.distance_walked += distance_to_travel
                    if scenario.grid[next_pos] != Scenario.NAME2ID['TARGET']:
                        scenario.grid[next_pos] = Scenario.NAME2ID['PEDESTRIAN']
                    elif scenario.grid[next_pos] == Scenario.NAME2ID['TARGET']:
                        self.finished = True
                    scenario.grid[self._position] = Scenario.NAME2ID['EMPTY']
                    self._position = next_pos
                else:
                    self.accumulated_distance += available_distance
                    available_distance = 0
            else:
                break
        self.ticks += 1
        return moved
