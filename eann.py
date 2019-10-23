# based on https://lethain.com/genetic-algorithms-cool-name-damn-simple/

import os
import pandas as pd
import pickle as pkl
import random
from keras import backend as K
from mlp import MLP


class Network:
    def __init__(self, p_space, config=None):
        """Initializes a network as an evolution population member.

        Params:
            p_space:    possible parameters for the networks'
                        configurations
            config:     a given configuration for the network.
                        if None, randomly initialize the network
        """
        self.parameter_space = p_space
        self.fitness = None
        # initialize the network by sampling from parameter space
        if config:
            self.configuration = config
        else:
            self.configuration = self.sample()

    def sample(self):
        """Sample a configuration for this network from the parameter space.

        Returns:
            config: sampled configuration of the network
        """
        config = {}
        for parameter in self.parameter_space.keys():
            config[parameter] = random.choice(self.parameter_space[parameter])
        return config

    def mutate(self, mutate_chance):
        """Mutate this network with a probability of `mutate_chance`.

        Will change a randomly selected parameter of the network
        to a newly sampled value (from the parameter space) with the
        probability given by `mutate_chance`.

        Params:
            mutate_chance:  probability of mutation happening
        """
        # what to mutate
        mutation_param = random.choice(list(self.parameter_space.keys()))
        # how to mutate
        if random.random() < mutate_chance:
            new_val = random.choice(self.parameter_space[mutation_param])
            self.configuration[mutation_param] = new_val

    def build_and_train(self, runs=3):
        """Builds and trains the specified network.

        Parameters:
            runs:       how many training runs to average

        Returns:
            fitness:    fitness of the network
        """
        val_accs = []
        for i in range(runs):
            layers = self.configuration['layers']
            optimizer = self.configuration['optimizer']
            hid_act = self.configuration['hid_act']
            print('Training: {}, {}, [{}]'.format(
                optimizer, hid_act, ', '.join(map(str, layers))
            ))
            mlp = MLP(layers, 'mnist', optimizer, hid_act=hid_act)
            stats = mlp.train(3, 128)
            val_acc = stats['val_accuracy'][-1]
            val_accs.append(float(val_acc))
            K.clear_session()
            del mlp
        print('avg fitness', sum(val_accs) / 3.0)
        return sum(val_accs) / 3.0

    def get_fitness(self):
        """Returns the fitness of the network.

        If the network fitness hasn't been computed yet, it will build and
        train the network, compute the fitness and then return it.

        Returns:
            fitness:    fitness of the network
        """
        # if the fitness was already computed, just return it
        if self.fitness:
            return self.fitness

        # compute the fitness by training the network, then return it
        self.fitness = self.build_and_train()
        return self.fitness

    def breed(self, other, n_children=2, mutate_chance=0.2):
        """Returns new networks from this network breeded with
        the network given in `other`.

        Params:
            other:  another Network instance

        Returns:
            children:   list of `n_children` Network instances
        """
        mother = self.configuration
        father = other.configuration

        # breed children
        children = []
        for _ in range(n_children):
            child = {}
            for param in self.configuration.keys():
                child[param] = random.choice([mother[param], father[param]])
            children.append(child)

        # instantiate
        for i in range(n_children):
            net = Network(self.parameter_space, config=children[i])
            net.mutate(mutate_chance)
            children[i] = net

        return children

    def to_string(self):
        return "optimizer {}, hid_act {}, layers [{}]".format(
            self.configuration['optimizer'],
            self.configuration['hid_act'],
            ', '.join(map(str, self.configuration['layers']))
        )


class Evolution:
    def __init__(self, parameter_space, population_size, retain=0.4,
                 random_select=0.1, mutate_chance=0.2, dupes=False):
        self.parameter_space = parameter_space
        self.population = [
            Network(parameter_space) for _ in range(population_size)
        ]
        # keep track of total networks checked
        self.n_iters = 0
        # parameters
        self.dupes = dupes
        self.retain = retain
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        # logs
        self.steps = 0
        self.logs = {
            'step': [],
            'config': [],
            'accuracy': [],
            'avg_accuracy': [],
        }

    def remove_duplicates(self):
        new_population = []
        for i in range(len(self.population)):
            net = self.population[i]
            # generate new networks as long as this one is
            # already in the current population
            while net.to_string() in [n.to_string() for n in new_population]:
                net = Network(self.parameter_space)
            # add the network to the population
            new_population.append(net)
        self.population = new_population

    def step(self):
        # optionally remove duplicates
        if not self.dupes:
            self.remove_duplicates()

        # keep track of the average fitness
        n_already_trained = sum([1 for net in self.population if net.fitness])
        fitnesses = [net.get_fitness() for net in self.population]
        avg_fitness = float(sum(fitnesses)) / len(fitnesses)
        self.n_iters += len(self.population) - n_already_trained

        # sort the population by score, descendingly
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)

        # save logs
        self.steps += 1
        for net in self.population:
            self.logs['step'].append(self.steps)
            self.logs['config'].append(net.to_string())
            self.logs['accuracy'].append(net.get_fitness())
            self.logs['avg_accuracy'].append(avg_fitness)

        # parents are the best networks in the population
        cutoff = int(len(self.population) * self.retain)
        parents = self.population[:cutoff]

        # keep some candidates randomly
        candidates = self.population[cutoff:]
        n_random_cand = int(len(self.population) * self.random_select)
        parents += random.choices(candidates, k=n_random_cand)

        # figure out how many children we need
        n_children_needed = len(self.population) - len(parents)

        # start breeding children
        children = []

        while len(children) < n_children_needed:
            # get parents randomly
            idx_m, idx_d = tuple(random.choices(range(len(parents)), k=2))
            if idx_m == idx_d:
                continue
            mom = parents[idx_m]
            dad = parents[idx_d]

            # breed two babies
            babies = mom.breed(dad, n_children=2,
                               mutate_chance=self.mutate_chance)

            children += babies

        # make sure we didn't breed too many babies
        children = children[:n_children_needed]

        # set new population
        self.population = parents + children

        # log
        print('\nEvolution Step {}. Avg Fitness {:.2%}.\n'.format(
            self.steps, avg_fitness
        ))

    def run(self, n_iters):
        while self.n_iters < n_iters:
            self.step()
            self.save()
        print(self.n_iters)

    def save(self):
        filename = 'results/eann_{}dupes_step{}.csv'.format(
            '' if self.dupes else 'no',
            self.steps
        )
        while os.path.exists(filename):
            filename = filename.replace('.csv', '_.csv')
        try:
            search_result_df = pd.DataFrame(self.logs)
            search_result_df.to_csv(filename)
        except Exception:
            pkl.dump(self.logs, open(filename, 'wb'))
