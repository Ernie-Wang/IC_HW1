
import numpy as np
import random
import math

from benchmark import F6 as test

''' Constant variable '''
epsilon = 0
RATE = 20
K_best = 50

class GSA():

    def __init__(self, g_0, dim, num, rate, k, max_iter, u_bound, l_bound, func, end_thres):
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
        self.end_thres = end_thres                      # Threshold to terminate algorithm

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
        self.best_sofar = 1000000                       # Record best fitness value so far
        self.best_results = np.zeros((self.max_iter))                   # Fitness value of the agent
        self.best_results_so_far = np.zeros((self.max_iter))                   # Fitness value of the agent

    def evaluate(self):
        for i in range(self.N):
            result = self.func(self.X[i])
            if result != 0:
                self.fit[i] = -result


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
            tmp = (self.best * 1.00005) / self.worst
        for i in range(self.N):
            if self.fit[i] != self.worst:
                self.m[i] = (self.fit[i] - self.worst) / tmp
            else:
                self.m[i] = (0.0001) / tmp
    
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
        
        if self.best == -1 and self.worst == -1:
            i = 0
    
    def distance(self, a, b):
        return math.sqrt(np.sum(np.square(a-b)))
        # dis_2 = 0
        # for d in range(self.dim):
        #     tmp = a[d] - b[d]
        #     dis_2 = dis_2 + tmp * tmp
        # return math.sqrt(dis_2)
    

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
    
    def result(self, iteration):
        # self.tmp = np.zeros((self.N))                   # Fitness value of the agent
        # for i in range(self.N):
        #     self.tmp[i] = self.func(self.X[i])
        # print("Best_fitness: ", 1/self.best)
        # print("Total: ", self.tmp)

        sort_index = np.argsort(self.fit)
        sort_index = np.flip(sort_index)
        self.best_results[iteration] = self.func(self.X[sort_index[0]])
        if self.best_sofar > self.best_results[iteration]:
            self.best_sofar = self.best_results[iteration]
        self.best_results_so_far[iteration] = self.best_sofar
        upper = lower = self.best_results[iteration]
        # upper = self.best_results[iteration]
        # lower = self.best_results[iteration]
        for i in range(20):
            if upper < self.best_results[iteration - i]:
                upper = self.best_results[iteration - i]

            elif lower > self.best_results[iteration - i]:
                lower = self.best_results[iteration - i]

        # print("Best fitness: ",self.func(self.X[sort_index[0]]) )
        print("Best: ",self.X[sort_index[0]], "fitness: ", self.best_results[iteration])
        if(upper-lower) < self.end_thres:
            return True
        else:
            return False
        
        pass
      
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


    def algorithm(self):
        # Initial
        self.gsa_init()

        # iteration
        for iteration in range(self.max_iter):
            self.gsa_iter(iteration)
            end = self.result(iteration)
            if end:
                break


if __name__ == "__main__":
    f7 = GSA (g_0 = 10000, dim=30, num=50, rate=RATE, k=K_best, max_iter=2500, u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=1e-8)
    f7.algorithm()