from eann import Evolution
import time

# set up the search space
optimizer = ['sgd', 'rms']
n_layers = list(range(1, 11))
n_hidden = list(range(100, 1001, 50))
layers = [[hid for i in range(lay)] for hid in n_hidden for lay in n_layers]
hid_act = ['tanh', 'relu']

# create the search space
params_dist = dict(optimizer=optimizer, layers=layers, hid_act=hid_act)

# setup evolution
start_time = time.time()
population_size = 50
n_iters = 200
evolution = Evolution(params_dist, population_size, retain=0.4,
                      random_select=0.1, mutate_chance=0.2, dupes=False)
evolution.run(n_iters)

duration = time.time() - start_time
print('Total time: {:.3f}'.format(duration))
