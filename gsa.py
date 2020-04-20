
import numpy as np
import random
import math

from benchmark import F7 as test

''' Constant variable '''
epsilon = 0
RATE = 20
K_best = 50

class GSA():

    def __init__(self, g_0, dim, num, rate, k, max_iter, u_bound, l_bound, func):
        """ Initialize GSA object """
        self.G = g_0 + g_0 * rate                       # Gravity of the algorithm
        self.G_0 = g_0                                  # Initial Gravity of the algorithm
        self.dim = dim                                  # Searching dimension
        self.N = num                                    # Number of agent
        self.limit = 0.6*dim*num                        # Searching limit
        self.rate = rate                                # Decreasing rate of the gravity
        self.alpha = rate
        self.K = k                                      # K best
        self.max_iter = max_iter                        # Maximum iteration
        self.K_rate = round(max_iter / 2 /num)          # The decrease rate of K
        self.u_bound = u_bound                          # Upper bound
        self.l_bound = l_bound                          # Lower bound
        self.func = func                                # Benchmark function

        self.M = np.zeros((self.N))                     # Mess of the agent
        self.m = np.zeros((self.N))                     # Mess calculate by fitness
        self.Total_F = np.zeros((self.N, self.dim))     # Total force of every agent
        self.F = np.zeros((self.N, self.N, self.dim))   # Force of every agent to others
        self.X = np.zeros((self.N, self.dim))           # Position of the agent
        self.V = np.zeros((self.N, self.dim))           # Velocity
        self.A = np.zeros((self.N, self.dim))           # Acceleration
        self.best = 0                                   # The best value of fitness func.
        self.worst = 0                                  # The worst value of fitness func.
        self.fit = np.zeros((self.N))                   # Fitness value of the agent

        self.trial = np.zeros((self.N))                 # Food source try time
        self.bestx = np.zeros((self.dim))               # Global best position
        self.best = 1000                                # Global best fitness

    def evaluate(self):
        for i in range(self.N):
            result = self.func(self.X[i])
            if result != 0:
                self.fit[i] = 1 / abs(result)

    def update_v_x_i(self, i):
        for d in range(self.dim):
            self.V[i][d] = random.random() * self.V[i][d] + self.A[i][d]
            self.X[i][d] = self.X[i][d] + self.V[i][d]
    
    def update_v_x(self):
        for i in range(self.N):
            self.update_v_x_i(i)

    def update_G(self, iteration):
        # self.G = self.G - self.G * self.rate
        self.G = self.G_0 * math.exp(-self.alpha * iteration / self.max_iter)

    def update_M(self):
        total_mess = np.sum(self.m)
        for i in range(self.N):
            self.M[i] = self.m[i] / total_mess

    def update_m(self):
        tmp = (self.best - self.worst)
        if tmp == 0:
            tmp = (self.best + 1) / self.worst
        for i in range(self.N):
            if self.fit[i] != self.worst:
                self.m[i] = (self.fit[i] - self.worst) / tmp
            else:
                self.m[i] = (self.fit[i] + 0.0001 - self.worst) / tmp
    
    def update_A(self):
        for i in range(self.N):
            for d in range(self.dim):
                self.A[i][d] = self.Total_F[i][d] / self.M[i]

    def update_K(self, iteration):
        if self.K > 1 and iteration % self.K_rate == 0:
            self.K = self.K - 1

    def find_limit(self):
        self.best = self.fit[0]                             # The best value of fitness func.
        self.worst = self.fit[0]                            # The worst value of fitness func.
        for i in self.fit:
            if i > self.best:
                self.best = i
            elif i < self.worst:
                self.worst = i
    
    def distance(self, a, b):
        dis_2 = 0
        for d in range(self.dim):
            tmp = a[d] - b[d]
            dis_2 = dis_2 + tmp * tmp
        return math.sqrt(dis_2)
    

    def force_ijd(self, M_1, M_2, d, pre_cal):
        return pre_cal * (self.X[M_2][d] - self.X[M_1][d])

    def force_i(self, M_1):
        for M_2 in range(self.N):
            if M_2 > M_1:
                pre_cal = self.G * self.M[M_1] * self.M[M_2] / (self.distance(self.X[M_1], self.X[M_2]) + epsilon)
                for d in range(self.dim):
                    m2_to_m1_f = self.force_ijd(M_1, M_2, d, pre_cal)
                    self.F[M_1][M_2][d] = m2_to_m1_f
                    self.F[M_2][M_1][d] = -m2_to_m1_f
    
    def update_f(self):
        for i in range(self.N):
            self.force_i(i)

    def total_force_i(self, M_i, sort_index):
        for d in range(self.dim):
            self.Total_F[M_i][d] = 0
            for i in range(self.K):
                if sort_index[i] != M_i:
                    f = self.F[M_i][sort_index[i]][d]
                    self.Total_F[M_i][d] = self.Total_F[M_i][d] + random.random() * self.F[M_i][sort_index[i]][d]

    def update_total_f(self):
        sort_index = np.argsort(self.fit)
        sort_index = np.flip(sort_index)
        for i in range(self.N):
            self.total_force_i(i, sort_index)
    
    def result(self):
        # self.tmp = np.zeros((self.N))                   # Fitness value of the agent
        # for i in range(self.N):
        #     self.tmp[i] = self.func(self.X[i])
        # print("Best_fitness: ", 1/self.best)
        # print("Total: ", self.tmp)

        sort_index = np.argsort(self.fit)
        sort_index = np.flip(sort_index)

        print("Best: ",self.X[sort_index[0]], "fitness: ",self.func(self.X[sort_index[0]]) )
      
    def gsa_init(self):
        # Initialize food source for all employed bees
        for i in range(self.N):
            self.X[i] = np.random.uniform(self.l_bound,self.u_bound, (self.dim))
            self.fit[i] = self.func(self.X[i])

    '''
    One Iteration of the gsa algorithm
    '''
    def gsa_iter(self, iteration):
        # Evaluate the fitness
        self.evaluate()

        # Update G
        self.update_G(iteration)

        # find limit
        self.find_limit()

        # Calculate Mess
        self.update_m()
        self.update_M()
        self.update_K(iteration)

        # Calculate force
        self.update_f()
        self.update_total_f()

        # Calculate Acceleration
        self.update_A()

        self.update_v_x()


    def gsa_algorithm(self):
        # Initial
        self.gsa_init()

        # iteration
        for iteration in range(self.max_iter):
            self.gsa_iter(iteration)
            self.result()


if __name__ == "__main__":
    f7 = GSA (g_0 = 100, dim=30, num=50, rate=RATE, k=K_best, max_iter=2500, u_bound=30, l_bound=-30, func=test.func)
    f7.gsa_algorithm()