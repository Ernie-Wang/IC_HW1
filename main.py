import numpy as np
import csv
from matplotlib import pyplot as plt

from pso import PSO
from gsa import GSA
from abc_py import ABC
from benchmark import F8 as test

''' Constant variable '''

######## Global variable #########
RUNS = 50
AGENT_NUM = 50
ITER_KINDS = 2
ALGO = 3
ITER = [500, 2500]
RESULTS = np.zeros((ALGO, RUNS, ITER_KINDS, ITER[1]))                   # Store all the result for the whole runs
AVERAGE_RESULT = np.zeros((ALGO, ITER_KINDS, ITER[1]))                   # Store all the result for the whole runs

##################################

######### PSO variable ###########

##################################

######### GSA variable ###########
epsilon = 0
G_0 = 100
ALPHA = 20
K_best = 50
end_thres = 1e-5
##################################

######### ABC variable ###########

##################################
def write_file():
    filename = "./result_csv/Total_{func}.csv".format(func=test.name)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iter', 'PSO500', 'GSA500', 'ABC500', 'PSO2500', 'GSA2500', 'ABC2500'])
        for i in range(ITER[1]):
            if i < ITER[0]:
                writer.writerow([i+1, AVERAGE_RESULT[0][0][i], AVERAGE_RESULT[1][0][i], AVERAGE_RESULT[2][0][i], AVERAGE_RESULT[0][1][i], AVERAGE_RESULT[1][1][i], AVERAGE_RESULT[2][1][i]])
    pass

def plot_result():
    x1 = np.arange(0,  500,  1) 
    x2 = np.arange(0,  2500,  1) 
    plt.figure(1)
    # plt.subplot(3,  1,  1)  

    title = "{func}, ITER=500".format(func=test.name)
    plt.title(title) 
    plt.xlabel("iter") 
    plt.ylabel("fitness") 
    for i in range(ALGO):
      tmp = AVERAGE_RESULT[i][0].copy()
      tmp.resize(ITER[0])
      plt.plot(x1, tmp) 

    plt.figure(2)
    # plt.subplot(3,  1,  3)  

    title = "{func}, ITER=2500".format(func=test.name)
    plt.title(title) 
    plt.xlabel("iter") 
    plt.ylabel("fitness") 
    for i in range(ALGO):
      plt.plot(x2, AVERAGE_RESULT[i][1]) 
    plt.show()

if __name__ == "__main__":
    for run in range(RUNS):
        for kind in range(ITER_KINDS):
            
            ## Initial random variables, every algorithm has same initial
            arr = np.random.uniform(test.l_bound,test.u_bound, (AGENT_NUM, test.dim))

            #########   PSO   #########
            algo = PSO (dim=test.dim,num=AGENT_NUM,max_iter=ITER[kind], u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
            algo.pso_init(arr)
            algo.pso_iterator()

            # Resize the result to 2500
            tmp = algo.best_results.copy()
            tmp.resize(2500)
            RESULTS[0][run][kind] = tmp.copy()
            ###########################

            #########   GSA   #########
            algo = GSA (g_0 = G_0, dim=test.dim, num=AGENT_NUM, rate=ALPHA, k=K_best, max_iter=ITER[kind], u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
            algo.algorithm(arr)

            # Resize the result to 2500
            tmp = algo.best_results_so_far.copy()
            tmp.resize(2500)
            RESULTS[1][run][kind] = tmp.copy()
            ###########################

            #########   ABC   #########
            algo = ABC (dim=test.dim, num=AGENT_NUM, max_iter=ITER[kind], u_bound=test.u_bound, l_bound=test.l_bound, func=test.func, end_thres=end_thres)
            algo.abc_init(arr)
            algo.abc_iterator()

            # Resize the result to 2500
            tmp = algo.best_results.copy()
            tmp.resize(2500)
            RESULTS[2][run][kind] = tmp.copy()
            ###########################

    for algo in range(ALGO):
        for kind in range(ITER_KINDS):

            average = np.zeros((ITER[1])) 
            for run in range(RUNS):
                
                average = average + RESULTS[algo][run][kind]
            AVERAGE_RESULT[algo][kind] = average / RUNS
    write_file()
    plot_result()