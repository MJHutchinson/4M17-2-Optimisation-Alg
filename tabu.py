from problem import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

DIM = 2

STM_SIZE = 7
MTM_SIZE = 4

INTENSIFY = 10
DIVERSIFY = 15
REDUCE = 25

GRID_PER_DIM = 10
GRID_WIDTH = int(2*LIM/GRID_PER_DIM)

START_DELTA = 10

FORCE_EXPLORE = False

EVALS = 100  # Number of evaluations of function for statistical averaging


class LastK(object):

    def __init__(self, k):
        self.list = []
        self.k = k

    def add(self, value):
        self.list.append(value.tolist())
        self.list = self.list[-self.k:]

    def is_in(self, point):
        return point.tolist() in self.list


class LargestK(object):

    def __init__(self, k):
        self.list = []
        self.k = k

    def add(self, value):
        if not value in self.list:
            self.list.append(value)
            self.list.sort(key=lambda x: x[0])
            self.list = self.list[:self.k]

    def get_best(self):
        return self.list[0]

    def get_position_mean(self):
        xs = [x[1] for x in self.list]
        return np.mean(xs, axis=0)


class GridTracker(object):

    def __init__(self, lim, dim, grid_per_dim):
        self.grid = np.zeros([grid_per_dim for dim in range(dim)])
        self.lim = lim
        self.dim = dim

        self.grid_width = int(2*lim/grid_per_dim)

    def add(self, point):
        grid_point = tuple([min(int((x + self.lim)/self.grid_width), GRID_PER_DIM-1) for x in point])
        self.grid[grid_point] += 1

    def sample(self):
        lim = self.grid.mean()

        while(True):
            point = np.random.uniform(-self.lim, self.lim, self.dim)
            grid_point = tuple([int((x + self.lim) / self.grid_width) for x in point])
            if self.grid[grid_point] < lim:
                return point

    def unexplored_areas(self):
        return (self.grid == 0).sum()


def check_neighbourhood(x_cen, d, stm):

    x_min = None
    f_min = np.inf

    evals = 0

    for i in range(len(x_cen)):
        x_test = np.copy(x_cen)
        x_test[i] = x_test[i] + d
        x_test = bound(x_test)

        if stm.is_in(x_test):
            continue

        f_test = f(x_test)
        evals += 1

        if f_test < f_min:
            x_min = x_test
            f_min = f_test

    for i in range(len(x_cen)):
        x_test = np.copy(x_cen)
        x_test[i] -= d
        x_test = bound(x_test)

        if stm.is_in(x_test):
            continue

        f_test = f(x_test)
        evals += 1

        if f_test < f_min:
            x_min = x_test
            f_min = f_test

    if x_min is None:
        return x_cen, f(x_cen), evals
    else:
        return x_min, f_min, evals


def search():
    x_hist = []
    f_hist = []
    evals = 0

    ltm = GridTracker(LIM, DIM, GRID_PER_DIM)
    mtm = LargestK(MTM_SIZE)
    stm = LastK(STM_SIZE)

    x_init = np.random.uniform(-LIM, LIM, DIM)
    f_init = f(x_init)
    evals += 1

    d = START_DELTA

    x_base = x_init
    f_base = f_init

    x_best = x_init
    f_best = f_init

    iter_no_improv = 0

    while evals <= FUNC_EVAL_LIM:
        x_next, f_next, evl = check_neighbourhood(x_base, d, stm)
        evals += evl

        if f_next < f_base:

            x_patt = x_next + (x_next - x_base)
            f_patt = f(x_patt)
            evals += 1

            if f_patt < f_next:
                x_next = x_patt
                f_next = f_patt

        x_base = x_next
        f_base = f_next
        x_hist.append(x_base)
        f_hist.append(f_base)

        stm.add(x_next)
        mtm.add((f_next, x_next.tolist()))
        ltm.add(x_next)

        if f_base < f_best:
            x_best = x_base
            f_best = f_base
            iter_no_improv = 0
        else:
            iter_no_improv += 1

            if iter_no_improv == INTENSIFY:
                x_base = mtm.get_position_mean()
                f_base = f(x_base)
                evals += 1
                x_hist.append(x_base)
                f_hist.append(f_base)

            if iter_no_improv == DIVERSIFY:
                x_base = ltm.sample()
                f_base = f(x_base)
                evals += 1
                x_hist.append(x_base)
                f_hist.append(f_base)
                # print(f'diversified to {x_base}')
                if FORCE_EXPLORE & ltm.unexplored_areas():
                    iter_no_improv = INTENSIFY

            if iter_no_improv == REDUCE:
                d = d/2
                x_base = x_best
                f_base = f_best
                x_hist.append(x_base)
                f_hist.append(f_base)
                iter_no_improv = 0

    return x_best, f_best, x_hist, f_hist


def experiment(output = False, plot = False):

    if output:
        print('\n'   + ('*' * 30))

        print('\n Running experiment with parameters:')
        print(f'\t DIM = {DIM}')
        print(f'\t STM_SIZE = {STM_SIZE}')
        print(f'\t MTM_SIZE = {MTM_SIZE}')
        print(f'\t INTENSIFY = {INTENSIFY}')
        print(f'\t DIVERSIFY = {DIVERSIFY}')
        print(f'\t REDUCE = {REDUCE}')
        print(f'\t GRID_PER_DIM = {GRID_PER_DIM}')
        print(f'\t GRID_WIDTH = {GRID_WIDTH}')
        print(f'\t START_DELTA = {START_DELTA}')
        print(f'\t FORCE_EXPLORE = {FORCE_EXPLORE}')
        print('\n')

    x_best_hist = []
    f_best_hist = []

    for i in range(EVALS):
        x_best, f_best, x_hist, f_hist = search()

        x_best_hist.append(x_best)
        f_best_hist.append(f_best)

    x_best_hist = np.array(x_best_hist)
    f_best_hist = np.array(f_best_hist)

    if output:
        print(f'Lowest Minimum in current run found at: {x_best_hist[f_best_hist.argmin()]}, of value: {f_best_hist.min()}')
        print(f'Average Minimum found: {f_best_hist.mean()}, with standard deviation: {f_best_hist.std()}')


    if plot:
        if DIM == 2:
            plot_points(x_best_hist.T)

    if output:
        print('\n' + ('*' * 30))

    return x_best_hist, f_best_hist


if __name__ == '__main__':

    START_DELTA = 10

    FORCE_EXPLORE = False
    _,_,x,_ = search()
    plot_points(np.array(x).T)

    FORCE_EXPLORE = True
    _, _, x, _ = search()
    plot_points(np.array(x).T)

    #####################
    # Forced exploration comparison
    # FORCE_EXPLORE = False
    # start_deltas = [100, 50, 10, 5, 1]
    # f_results = []
    # for START_DELTA in tqdm(start_deltas):
    #     x_run, f_run = experiment(False)
    #     f_results.append(f_run)
    #
    # plt.figure()
    # plt.boxplot(np.array(f_results).T)
    # plt.xticks(range(1, len(start_deltas)+1), start_deltas)
    # plt.xlabel('Initial $\delta$')
    # plt.ylabel('Statistics of minimums found')
    # plt.title('Evaluation of different initial $\delta$ for no forced exploration')
    # plt.show()
    #
    # FORCE_EXPLORE = True
    # start_deltas = [100, 50, 10, 5, 1]
    # f_results = []
    # for START_DELTA in tqdm(start_deltas):
    #     x_run, f_run = experiment(False)
    #     f_results.append(f_run)
    #
    # plt.figure()
    # plt.boxplot(np.array(f_results).T)
    # plt.xticks(range(1, len(start_deltas) + 1), start_deltas)
    # plt.xlabel('Initial $\delta$')
    # plt.ylabel('Statistics of minimums found')
    # plt.title('Evaluation of different initial $\delta$ for forced exploration')
    # plt.show()
    #####################


