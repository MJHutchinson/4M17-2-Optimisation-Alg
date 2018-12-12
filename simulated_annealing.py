from problem import *
import matplotlib.pyplot as plt

DIM = 2  # Problem dimensions
INITIAL_T_PROB = 0.8  # Initial probability that solution should be accepted
MAX_CHANGE = 100  # Maximum change of variable in single step
T_EST_SAMPLES = 100  # Number of samples to use to find initial T

L_K = 100  # Markov chain length
ETA_MIN = 0.6  # Acceptance proportion of chain to finish anneal

ANNEALING = 'Kirkpatrick'   # 'Kirkpatrick', 'Huang'
ALPHA = 0.95  # Kirkpatrick et al. [1982] Exponential cooling

RESET_THRESHOLD = 200

METHOD = 'Simulated Annealing'


def estimate_initial_temperature(num_evals):
    x_best = None
    f_best = np.inf

    x_curr = np.array([-400, -400]) # np.random.uniform(-LIM, LIM, DIM)
    f_curr = f(x_curr)

    x_best = x_curr
    f_best = f_curr

    df_buffer = []

    for i in range(num_evals):
        x_new = gen_x_simple(x_curr)
        f_new = f(x_new)

        if f_new - f_curr > 0:
            df_buffer.append(f_new - f_curr)

        x_curr = x_new
        f_curr = f_new

        if f_curr < f_best:
            x_best = x_curr
            f_best = f_curr

    df_buffer = np.array(df_buffer)
    df_mean = df_buffer.mean()

    T_mean = - df_mean / np.log(INITIAL_T_PROB)
    T_std = df_buffer.std()

    return T_mean, T_std, x_best, f_best


def gen_x_simple(x):
    C = MAX_CHANGE * np.ones(DIM)
    u = np.random.uniform(-1, 1, DIM)
    x_next = x + C * u  # element-wise multiplication

    while (x_next > LIM).any() | (x_next < -LIM).any():
        u = np.random.uniform(-1, 1, DIM)
        x_next = x + C * u  # element-wise multiplication
    return x_next


def check_accept_solution(df, T):
    if df < 0:
        return True
    else:
        p = np.exp(-1. * df / T)
        if np.random.random() < p:
            return True
        else:
            return False


def search():
    evals = 0

    T_mean, T_std, x_start, f_start = estimate_initial_temperature(T_EST_SAMPLES)

    T = T_mean
    x_best = np.random.uniform(-LIM, LIM, DIM)  # x_start
    f_best = f(x_best)  # f_start
    x_curr = x_best
    f_curr = f_best
    evals += T_EST_SAMPLES

    l_curr = 0
    eta_curr = 0
    f_acc_curr = []
    evals_since_best = 0
    found_best_curr = False

    x_hist = []
    f_hist = []
    T_hist = []
    acc_hist = []
    best_hist = []

    acc = 0

    while evals < FUNC_EVAL_LIM:
        # if (evals % 100) == 0:print(evals)

        if eta_curr > (L_K * ETA_MIN) or l_curr > L_K:

            ## Annealing logic
            if ANNEALING == 'Kirkpatrick':
                T = T * ALPHA
            elif ANNEALING == 'Huang':
                if len(f_acc_curr) > 1:
                    sigma = np.std(f_acc_curr)
                    alpha = max(0.5, np.exp( -(0.7 * T)/sigma))
                else:
                    alpha = 0.5
                T = T * alpha
            else:
                raise ValueError('Please select a valid Annealing method')

            l_curr = 0
            eta_curr = 0
            f_acc_curr = []

        x_next = gen_x_simple(x_curr)

        if (x_next > LIM).any() | (x_next < -LIM).any():
            continue

        f_next = f(x_next)
        evals += 1
        l_curr += 1

        x_hist.append(x_next)
        f_hist.append(f_next)
        T_hist.append(T)

        if check_accept_solution(f_next - f_curr, T):
            x_curr = x_next
            f_curr = f_next

            eta_curr += 1
            acc += 1
            acc_hist.append(1)
            f_acc_curr.append(f_curr)

            if f_next < f_best:
                x_best = x_curr
                f_best = f_curr
                best_hist.append(1)
                found_best_curr = True
                evals_since_best = 0
            else:
                best_hist.append(0)
                evals_since_best += 1

            if evals_since_best > RESET_THRESHOLD:
                x_curr = x_best
                f_curr = f_best

                # eta_curr = 0
                # l_curr = 0
                f_acc_curr = []
                evals_since_best = 0
                found_best_curr = False
                # print(f'reset at {evals}')

        else:
            acc_hist.append(0)
            best_hist.append(0)

    return x_best, f_best, (x_hist, f_hist, T_hist, acc_hist, best_hist)


def run():
    pass


if __name__ == '__main__':
    x_best, f_best, (x_hist, f_hist, T_hist, acc_hist, best_hist) = search()

    x_hist = np.array(x_hist)
    f_hist = np.array(f_hist)
    T_hist = np.array(T_hist)
    acc_hist = np.array(acc_hist)
    best_hist = np.array(best_hist)

    acc_hist = acc_hist == 1
    best_hist = best_hist == 1

    plt.plot(f_hist)
    plt.show()

    plt.plot(acc_hist)
    plt.show()

    plt.plot(T_hist)
    plt.show()

    plt.plot(f_hist[acc_hist])
    plt.plot(best_hist[acc_hist] * 100)
    plt.show()

    plot_points(x_hist.transpose())
    plot_walk(x_hist[best_hist].transpose())
    plot_walk(x_hist[acc_hist].transpose())

    x_acc = x_hist[acc_hist]
    acc = acc_hist.sum()

    plot_points(x_acc[int(0.7 * acc):].transpose())
    plot_points(x_acc[int(0.99 * acc):].transpose())
