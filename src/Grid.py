import time
import math
from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, height, width):
        # Set seed for random number generator
        self.rng = np.random.default_rng(0)

        # Create grid
        self.height = height
        self.width = width
        self.grid = self.make_grid()

        # Initialise start and end cells
        self.start_cell = (0, 0)
        self.end_cell = (height - 1, width - 1)

    # Create a 2d array of given height and width, and fill it with random numbers between 0 and 9
    def make_grid(self):
        # Create 2d array filled with 0
        grid = np.zeros((self.height, self.width), dtype=int)
        # Change each 0 to a random number between 0 and 9
        for y in range(self.height):
            for x in range(self.width):
                grid[y][x] = self.rng.integers(0, 10)

        return grid

    # Display the current state of the grid using pyplot
    def display_grid(self, current_cells):
        # Create array which will store colour of cells
        cell_colour = np.ones((self.height, self.width, 3))  # Start with an array of white

        # Set colour to cells
        cell_colour[self.start_cell] = 0.35, 0.78, 0.28
        cell_colour[self.end_cell] = 0.68, 0.07, 0.07
        for cell in current_cells:
            cell_colour[cell] = 0.94, 0.79, 0.24

        plt.figure(figsize=(self.height * .7, self.width * .7))
        plt.imshow(cell_colour)

        # Plot the numbers in the grid
        for (x, y), value in np.ndenumerate(self.grid):
            plt.text(y, x, value, ha='center', va='center', fontsize=15)

        # Remove axis and tick labels
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.tick_params(tick1On=False)

        # Display grid in the right position
        plt.xticks(np.arange(-.5, self.width + 1, 1.0))
        plt.yticks(np.arange(-.5, self.height + 1, 1.0))
        ax.margins(0)
        plt.grid(color='k', linewidth=2)

        plt.show()

    # Return a list of all the neighbours indexes of a given cell
    def get_cell_neighbours(self, cell):
        neighbours_indexes = []

        x = cell[1]
        y = cell[0]

        # Return true if each coordinate is within the bounds of the grid
        def is_in_bounds(c):
            return 0 <= c[0] < self.height and 0 <= c[1] < self.width

        # Add all neighbours to the array
        if is_in_bounds((y - 1, x - 1)):
            neighbours_indexes.append((y - 1, x - 1))
        if is_in_bounds((y - 1, x)):
            neighbours_indexes.append((y - 1, x))
        if is_in_bounds((y - 1, x + 1)):
            neighbours_indexes.append((y - 1, x + 1))
        if is_in_bounds((y, x + 1)):
            neighbours_indexes.append((y, x + 1))
        if is_in_bounds((y + 1, x + 1)):
            neighbours_indexes.append((y + 1, x + 1))
        if is_in_bounds((y + 1, x)):
            neighbours_indexes.append((y + 1, x))
        if is_in_bounds((y + 1, x - 1)):
            neighbours_indexes.append((y + 1, x - 1))
        if is_in_bounds((y, x - 1)):
            neighbours_indexes.append((y, x - 1))

        return neighbours_indexes

    # Return a list of cell indexes that form a path from the start cell to the end cell
    def find_path(self):
        path = []

        previous_cell = self.start_cell
        current_cell = self.start_cell

        # Loop until end cell is reached
        while current_cell != self.end_cell:
            # Add the current cell to the path and get all its neighbours
            path.append(current_cell)
            neighbours = self.get_cell_neighbours(current_cell)

            # Initialise next cell with first neighbour in list
            next_cell = neighbours[0]

            # Iterate through all the neighbours
            for neighbour in neighbours:

                # If neighbour is the end cell, make the current cell the neighbour and break out of loop
                if neighbour == self.end_cell:
                    current_cell = neighbour
                    break

                # Assign the next cell and the neighbour costs using their distance to the end and their cost
                next_cell_cost = math.dist(next_cell, self.end_cell) - (10 - self.grid[next_cell]) / 3
                neighbour_cost = math.dist(neighbour, self.end_cell) - (10 - self.grid[neighbour]) / 3

                # If the neighbour's cost is lower than the next cell cost and the neighbour is not the
                # previous cell, make the next cell the neighbour
                if neighbour_cost < next_cell_cost and neighbour != previous_cell:
                    next_cell = neighbour

            # If current cell is the end cell, break out of loop
            if current_cell == self.end_cell:
                break

            # Make previous cell the current cell and the current cell the next cell
            previous_cell = current_cell
            current_cell = next_cell

        # Remove start cell from path, add the end cell and return it
        path.pop(0)
        path.append(current_cell)

        return path

    # Use dijkstra algorithm to return a list of cell indexes that form the shortest path from
    # the start cell to the end cell
    def dijkstra(self):
        path = []

        # Add start cell to priority queue with a cost of 0
        queue = PriorityQueue()
        queue.put((0, self.start_cell))

        costs = {self.start_cell: 0}
        visited_cells = set()
        previous_cells = {}

        # Iterate while queue is not empty
        while queue.queue:
            # Pop the lowest cost cell from queue and get its neighbours
            cost, current_cell = queue.get()
            neighbours = self.get_cell_neighbours(current_cell)

            # If cell has been visited continue
            if current_cell in visited_cells:
                continue

            # Add cell to visited cells
            visited_cells.add(current_cell)

            # Iterate through all the neighbours
            for neighbour in neighbours:
                # If neighbour has been visited continue
                if neighbour in visited_cells:
                    continue

                # Get the cost to neighbour by adding the current cell's cost and the neighbour's cost
                cost_to_neighbour = cost + self.grid[neighbour]

                # If the cost to neighbour is less than their current cost, update their cost and add the
                # neighbour to the queue, also set the neighbour's previous cell to the current cell
                if cost_to_neighbour < costs.get(neighbour, float('inf')):
                    queue.put((cost_to_neighbour, neighbour))
                    previous_cells[neighbour] = current_cell
                    costs[neighbour] = cost_to_neighbour

        # Create path by starting from the end cell and getting all the previous cells until the start
        # cell is reached
        path.append(self.end_cell)
        previous_cell = previous_cells[self.end_cell]
        while previous_cell != self.start_cell:
            path.insert(0, previous_cell)
            previous_cell = previous_cells[previous_cell]

        return path

    # Use display_grid to display a given path being traversed, waiting an amount of time dependent
    # on the cell's cost after moving to it
    def traverse_path(self, path):
        for cell in path:
            self.display_grid([cell])
            time.sleep(0.25 * self.grid[cell])

    # Return the sum of all the cells' costs of a path
    def get_path_length(self, path):
        length = 0

        for cell in path:
            length += self.grid[cell]

        return length


grid = Grid(4, 8)

grid.traverse_path(grid.find_path())
grid.display_grid(grid.find_path())
print(grid.get_path_length(grid.find_path()))

grid.traverse_path(grid.dijkstra())
grid.display_grid(grid.dijkstra())
print(grid.get_path_length(grid.dijkstra()))
