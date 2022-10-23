"""
This file implements a Mixed Linear Program (MLP) that can solve instances
regarding the offline "Telescope Scheduling" problem. It does so by reading in
a file of a specific form, of which an example will be provided at the end of
this document header. The standard usage of this program is done via the
command line interface, writing: python3 MLP.py problemfile.txt outputfile.txt
For more information on MLP's, see:
https://en.wikipedia.org/wiki/Integer_programming.

Example input file and structure:
First the number of images is provided, followed by their sizes, a new linefor each image size. After that, the number of blocks/unavailable windows is provided, followed by their starting times and their duration, a new line for each block. Example:
6
3.14
5
6
12.21
0.97
2.1
3
1, 1
6, 2.01
14, 7.6
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Tuple
import sys

def load_instance(path: str) -> Tuple[List, List, List, List]:
    """Load a test instance from a given path.

	Args:
		path: The path containing the test instance

    Returns:
		The images, the starting times of the blocks and their duration.
		Some instances have the proposed solution after the last block,
		which this function also returns.
	"""

    with open(path, 'r') as instance:
        data = instance.read().splitlines()

    img_count = int(data.pop(0))
    s = [float(data.pop(0)) for _ in range(img_count)]

    block_count = int(data.pop(0))

    t, l = [], []
    for _ in range(block_count):
        t_i, l_i = data.pop(0).split(',')
        t.append(float(t_i))
        l.append(float(l_i))

    d = [float(data.pop(0)) for _ in range(img_count)]

    return s, t, l, d



def write_solution(model: gp.Model, in_path: str, out_path:str) -> None:
    """Writes the solution given by the model to the out_path

	Args:
		model: The model that solved the instance containing the parameters
		in_path: The path to the problem instance
		out_path: The path specifying where to write the solution
	"""

    with open(in_path, 'r') as problem, open(out_path, 'w') as solution:
		# First copy the problem input
        data = problem.read().splitlines()

        num_images = int(data[0])

        # These are the lines on which the number of images and intervals are
        for _ in range(num_images + int(data[num_images + 1]) + 2):
            solution.write(f"{data.pop(0)}\n")

        # Second write the score
        solution.write(f"{str(round(model.getVarByName('T').X, 3))}\n")

        # Third write the start times of each image
        for i in range(num_images):
            solution.write(f"{str(round(model.getVarByName(f'x_{i}').X, 3))}\n")


def solve_instance(in_path: str, out_path: str, verbose=False) -> gp.Model:
    """Solves an instance specified by the in_path and writes it to out_path

	Args:
		in_path: The path to the problem instance
		out_path: The path specifying where to write the solution
		verbose: Whether to print the models solution process

	Returns:
		model: The model that solved the problem instance"""

    # Create the model and store it referencing its problem input
    model = gp.Model(f"telescope-lp-{in_path}")

    # Silent or verbose output of model optimization
    model.Params.LogToConsole = verbose

    # Define the constants from the problem input, formally s, t, and l
    img_sizes, block_times, block_lengths, _ = load_instance(in_path)

    # Define the decision variables
    # The time when all images have been sent, formally T
    end_time = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="T")

    # Construct the list with the startingtime for each image, formally x
    img_times = []
    for i in range(len(img_sizes)):
        # The default upperbound is infinity
        img_times.append(model.addVar(vtype=GRB.CONTINUOUS,
		                                   lb=0,
										   name=f"x_{i}"))

    # Construct the array to avoid overlapping start and end times
	# Formally y
    img_overlap = []
    for i in range(len(img_sizes)):
        row = []
        for j in range(len(img_sizes)):
            # An image cannot be sent before its own starting time
            # Thus, only create binary variables for cases when i != j
            if i != j:
                row.append(model.addVar(vtype=GRB.BINARY,
				                        name=f"y_{i},{j}"))

            # Ensure the matrix img_overlap is square
            else:
                row.append("i equals j")

        img_overlap.append(row)

    # Convert to numpy array for more convenient indexing
    img_overlap = np.array(img_overlap)

    # Construct the array for restricting the image starting times
	# Formally z
    block_overlap = []
    for i in range(len(img_sizes)):
        row = []
        for k in range(len(block_times)):
            row.append(model.addVar(vtype=GRB.BINARY, name=f"z_{i},{k}"))

        block_overlap.append(row)

    block_overlap = np.array(block_overlap)

    # Constraints
    for i in range(len(img_sizes)):
        # Every image must be scheduled before the end time
        # This ensures the images are sent as early as possible
        # Formally x_i + s_i ≤ T
        model.addConstr(img_times[i] + img_sizes[i] <= end_time,
		                name="All images must be sent before end time")
        for j in range(len(img_sizes)):
            if i != j:
                # Every image must either be sent before an image
				# XOR be sent after it
                # Formally y_ij * (x_i + s_i) ≤ x_j
                model.addConstr(img_overlap[i, j] * (img_times[i] +
				                img_sizes[i]) <= img_times[j],
								name="Finish sending image before next image")

                # Formally x_i ≥ (1 - y_ij) * (x_j + s_j)
                model.addConstr(img_times[i] >= (1 - img_overlap[i, j]) *
				                (img_times[j] + img_sizes[j]),
				                name="Send image after previous image finished")

            else:
                continue

        for k in range(len(block_times)):
            # Every image must either be sent before an unavailable time window
            # Xor must be sent after the end of an unavailable time window
            # Formally, z_ik * (x_i + s_i) ≤ t_k
            model.addConstr(block_overlap[i, k] * (img_times[i] +
			                img_sizes[i]) <= block_times[k],
							name="Finish sending image before every block")

            # Formally, x_i ≥ (1 - z_ik) * (t_k + l_k)
            model.addConstr(img_times[i] >= (1 - block_overlap[i, k]) *
			                (block_times[k] + block_lengths[k]),
							name="Image must start sending after every block")

    # Optimization function, minimze the end time
    # Formally min T
    model.setObjective(end_time, GRB.MINIMIZE)

    # Solve the Mixed Linear Program
    model.optimize()

    # Write the solution to the solution path
    write_solution(model, in_path, out_path)

    return model

if __name__ == "__main__":
	model = solve_instance(sys.argv[1], sys.argv[2])
