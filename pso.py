""" Python PSO """
import random
import numpy as np

from benchmark import F7 as test

class PSO():
    def __init__(self, dim, num, max_iter, u_bound, l_bound, func):
        """ Initialize PSO object """
        self.dim = dim                                # Searching dimension
        self.num = num                                # Number of particle
        self.max_iter = max_iter                      # Maximum iteration
        self.X = np.zeros((self.num, self.dim))       # Particle position
        self.V = np.zeros((self.num, self.dim))       # Particle velocity
        self.pbest = np.zeros((self.num, self.dim))   # Pbest position
        self.pbest_v = np.zeros(self.num)             # Pbest value
        self.gbest = np.zeros((1, self.dim))          # Gbest position
        self.gbest_v = 1000                           # Gbest value
        self.u_bound = u_bound                        # Upper bound
        self.l_bound = l_bound                        # Lower bound
        self.func = func                              # Benchmark function
        self.best_results = np.zeros((self.max_iter))                   # Fitness value of the agent

    def pso_init(self):
        """ Initialize particle attribute, best position and best value """
        for n in range(self.num):
            for d in range(self.dim):
                self.X[n][d] = random.uniform(self.l_bound,self.u_bound)
                self.V[n][d] = random.uniform(-1,1)
            self.pbest[n] = self.X[n].copy()
            self.pbest_v[n] = self.func(self.pbest[n])

            if self.pbest_v[n] < self.gbest_v:
                self.gbest_v = self.pbest_v[n]
                self.gbest = self.pbest[n].copy()

    def pso_iterator(self):
        """ Iteration """
        for ite_idx in range(self.max_iter):
            print("Iteration: {ite}, best is {best}".format(ite=ite_idx+1, best=self.gbest_v))

            # Particle iterator, update best value
            for part in range(self.num):
                test_tmp = self.func(self.X[part])
                # Update local attribute
                if test_tmp < self.pbest_v[part]:
                    self.pbest[part] = self.X[part].copy()
                    self.pbest_v[part] = test_tmp

                    # Update global attribute
                    if test_tmp < self.gbest_v:
                        self.gbest = self.X[part].copy()
                        self.gbest_v = test_tmp
            
            # Update particle position and velocity
            r1 = np.random.uniform(size=(self.num, self.dim))
            r2 = np.random.uniform(size=(self.num, self.dim))
            self.V = self.V*random.uniform(0.2,0.6) + 2*r1*(self.pbest-self.X) + 2*r2*(self.gbest-self.X)
            tmp_X = self.X + self.V
            tmp_X = np.where(tmp_X > self.u_bound, self.u_bound, tmp_X)
            tmp_X = np.where(tmp_X < self.l_bound, self.l_bound, tmp_X)
            self.X = tmp_X.copy()

            self.best_results[ite_idx] = self.gbest_v

if __name__ == "__main__":
    a = PSO (dim=30,num=50,max_iter=2500, u_bound=test.u_bound, l_bound=test.l_bound, func=test.func)
    a.pso_init()
    a.pso_iterator()