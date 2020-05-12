import lspi
from quadcopterpolicy import QuadcopterPolicy
from lspi.basis_functions import RadialBasisFunction
from quadcopterdomain import QuadcopterDomain
import random
from lspi.solvers import LSTDQSolver
from lspi import lspi
import numpy as np
import pickle
import time



def collect_samples(quad_domain, quad_policy):
    sample_data = []
    num_iterations = 1000
    target_list = [[1,1,1], [1,1,0], [1,0,1], [1,1,2], [1,-1,4],[-1,-1,2],[-1,1,4]]
    quad_domain.reset(np.array([1,0,4,0,0,0]))
    quad_domain.set_target_state([1,0,0])
    c = 0
    for i in range(num_iterations):
        action = random.randint(0,quad_domain.num_actions()-1) #quad_policy.select_action(quad_domain.current_state()) #random.randint(quad_domain.num_actions)
        print(action)
        sample = quad_domain.apply_action(action)
        sample_data.append(sample)
        if i % 200 == 0:
            quad_domain.set_target_state(target_list[c % 7])
            c += 1

    return sample_data



if __name__ == "__main__":


    quad_domain = QuadcopterDomain()
    num_actions = quad_domain.num_actions()
    #print(num_actions)
    mean_bf = [np.random.uniform(0,1, size = (6,))]
    #print(mean_bf, '************')
    basis_func = RadialBasisFunction(mean_bf, 0.5, num_actions)
    quad_policy = QuadcopterPolicy(basis_func)
    
    sample_data = collect_samples(quad_domain, quad_policy)
    print(sample_data[0])
    solver = LSTDQSolver()
    start = time.clock()
    new_policy = lspi.learn(sample_data, quad_policy, solver)
    print('Done!', (time.clock() - start))


    with open('weights.pickle', 'wb') as weights_file:
        pickle.dump(new_policy.weights, weights_file)
    



