import lspi
from quadcopterpolicy import QuadcopterPolicy
from lspi.basis_functions import RadialBasisFunction
from quadcopterdomain import QuadcopterDomain
import random
from lspi.solvers import LSTDQSolver
from lspi import lspi
import numpy as np
from gui import GUI
import math
import time
import pickle



def start_landing(init_position, target_position, weights):
    quad_domain = QuadcopterDomain()
    num_actions = quad_domain.num_actions()
    
    mean_bf = [np.random.uniform(0,1, size = (6,))]
    
    basis_func = RadialBasisFunction(mean_bf, 0.5, num_actions)
    quad_policy = QuadcopterPolicy(basis_func, weights=weights)

    gui_object = GUI(quad_domain.quad_dict)

    threshold_err = 0.01
    distance_err = distance(target_position, init_position)
    quad_domain.reset(np.array([1,0,4,0,0,0]))


    time_limit = 15*60 # 15min
    
    time_elapsed = 0
    start_time = time.clock()
    while distance_err > threshold_err and time_elapsed <= time_limit:
        action = quad_policy.best_action(quad_domain.current_state())
        quad_domain.apply_action(action)
        new_position = quad_domain.quad.get_position(quad_domain.key)
        distance_err = distance(target_position, new_position)
        print(new_position)
        #for i in range(300):
        gui_object.quads['q1']['position'] = [new_position[0], new_position[1], new_position[2]]
        gui_object.quads['q1']['orientation'] = quad_domain.quad.get_orientation(quad_domain.key)
        gui_object.update()
        
        time_elapsed = time.clock() - start_time



def distance(p1, p2):
    x_diff = p1[0] - p2[0]
    y_diff = p1[1] - p2[1]
    z_diff = p1[2] - p2[2]
    dist = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
    return dist


if __name__ == "__main__":
    init_position = [1,0,4]
    target_position = [1,0,0]
    weights = None
    with open('weights.pickle', 'rb') as weight_file:
        weights = pickle.load(weight_file)
    
    if weights is not None:
        start_landing(init_position, target_position, weights)