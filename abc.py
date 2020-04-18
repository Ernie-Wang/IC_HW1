import numpy as np
import random

from benchmark import F7 as test

class ABC():

    def __init__(self, dim, num, max_iter, u_bound, l_bound, func):
        """ Initialize ABC object """
        self.SN = num                                 # Number of onlooker bees / enployed bees
        self.dim = dim                                # Searching dimension
        self.limit = 0.6*dim*num                      # Searching limit
        self.max_iter = max_iter                      # Maximum iteration
        self.u_bound = u_bound                        # Upper bound
        self.l_bound = l_bound                        # Lower bound
        self.func = func                              # Benchmark function

        self.X = np.zeros((self.SN, self.dim))        # Food source position
        self.fit = np.zeros((self.SN))                # Food source fitness
        self.trial = np.zeros((self.SN))              # Food source try time
        self.bestx = np.zeros((self.dim))             # Global best position
        self.best = 1000                              # Global best fitness
      
    def abc_init(self):
        # Initialize food source for all employed bees
        for i in range(self.SN):
            self.X[i] = np.random.uniform(self.l_bound,self.u_bound, (self.dim))
            self.fit[i] = self.func(self.X[i])


    def abc_iterator(self):
        # Iteration
        for ite_idx in range(self.max_iter):
            print("Iteration: {ite}, best is {best}".format(ite=ite_idx+1, best=self.best))

            # 1. Send employed bees to the new food source
            for i in range(self.SN):
                # Random select a source to change but noice that j!= i
                j = random.choice([n for n in range(self.SN) if i != n])
                # Random select a dimension to change
                k = random.choice(range(0,self.dim))
                # Employed bee generate new position in the neighborhood of its present food source
                tmp_pos = self.X[i]
                change_pos = self.X[i][k] + random.uniform(-1,1) * (self.X[i][k] - self.X[j][k])
                tmp_pos[k] = change_pos
                tmp_pos = np.where(tmp_pos > self.u_bound, self.u_bound, tmp_pos)
                tmp_pos = np.where(tmp_pos < self.l_bound, self.l_bound, tmp_pos)

                # Calculate new position fitness
                tmp_fit = self.func(tmp_pos)
                # Greedy selection to select food source
                if tmp_fit < self.fit[i]:
                    self.X[i] = tmp_pos
                else:
                    self.trial[i] = self.trial[i] + 1
                    

            # Update new food source fitness
            self.fit = np.apply_along_axis(self.func, axis=1, arr=self.X)
            # Calculate Selection Probabilities
            p_source = self.fit/np.sum(self.fit)

            # 2. Send onlooker bees
            for i in range(self.SN):
                # Select Source Site by Roulette Wheel Selection)
                food_source = np.random.choice(self.SN, 1, p=p_source)
                # Random select a source to change but noice that j!= i
                j = random.choice([n for n in range(self.SN) if i != n])
                # Random select a dimension to change
                k = random.choice(range(0,self.dim))
                # Onlooker bee generate new position in the neighborhood of its present food source
                tmp_pos = self.X[i]
                change_pos = self.X[i][k] + random.uniform(-1,1) * (self.X[i][k] - self.X[j][k])
                tmp_pos[k] = change_pos
                tmp_pos = np.where(tmp_pos > self.u_bound, self.u_bound, tmp_pos)
                tmp_pos = np.where(tmp_pos < self.l_bound, self.l_bound, tmp_pos)
                # Calculate new food source fitness
                tmp_fit = self.func(tmp_pos)
                # Greedy selection to select food source
                if tmp_fit < self.fit[food_source]:
                    self.X[food_source] = tmp_pos
                else:
                    self.trial[food_source] = self.trial[food_source] + 1

            # 3. Send scout
            for i in range(self.SN):
                if self.trial[i] >= self.limit:
                    self.trial[i] = 0
                    self.X[i] = np.random.uniform(self.l_bound,self.u_bound, (self.dim))
                    self.fit[i] = self.func(self.X[i])
            
            # 4. Update best solution
            best_idx = np.argmin(self.fit)
            if self.fit[best_idx] < self.best:
                self.best = self.fit[best_idx]
                self.bestx = self.X[best_idx]

if __name__ == "__main__":
    f7 = ABC (dim=30, num=50, max_iter=2500, u_bound=600, l_bound=-600, func=test.func)
    f7.abc_init()
    f7.abc_iterator()