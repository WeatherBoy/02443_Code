import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import scipy
import collections


### >>>> BIG test & plots (ex01) <<<< ###
## Functions which are apparently needed in here:
def linear_congruental_generator(
    multiplier: int, shift: int, modulus: int, x0: int = None, break_point: int = None, modulus_overwrite: bool = False
) -> tuple[list[int], list[float]]:
    """
    Generates a list of random numbers using the linear congruental generator method.

    NOTE: random_nums and U will have length 'modulus' if the parameters follows the conditions
    of Theorem 1 Maximum Cycle Length.

    :param multiplier: The multiplier (a) used in the formula.
    :param shift: The shift (c) used in the formula.
    :param modulus: The modulus (M) used in the formula.
    :param x0: initial value (x0) used in the formula. If not provided, a random value will be used.
    :param break_point: The number of iterations to run the generator for.
    :param modulus_overwrite: If you specifically want to overwrite the modulus, set this to True.

    :return random_nums: a list of randomly generated numbers.
    :return U: a list of random numbers between 0 and 1.
    """

    assert type(modulus) is int, "Modulus must be an integer."
    assert modulus > 0, "Modulus must be greater than 0."

    if modulus < 10**8 and not modulus_overwrite:
        print("WARNING: Modulus is less than 10^8. This may result in a short cycle length.")
        modulus = np.random.randint(10**8, 10**10)  # <-- just some randomly big number
    if x0 is None:
        x0 = np.random.randint(0, modulus)

    if break_point is None:
        # NOTE: this makes zero sense
        if modulus_overwrite:
            # It i just that in this case we can't guarantee that modulus is very big, and we obviously want break_point to be greater than 1
            break_point = modulus // 10
        elif not modulus_overwrite:
            break_point = modulus // 1000

    random_nums = [x0]
    U = [x0 / modulus]
    random_nums_generated = 0
    while random_nums_generated < modulus and random_nums_generated < break_point:
        random_nums.append((multiplier * random_nums[-1] + shift) % modulus)
        U.append(random_nums[-1] / modulus)
        random_nums_generated += 1

    return random_nums, U


def chi_squared_test_U(U: list[float], num_bins: int) -> tuple[float, float, int]:
    """
    This function performs the chi-squared test on a list of random numbers.
    For uniform expected distribution.

    :param U: list of random numbers (uniformly distributed).
    :param num_bins: number of bins to divide the random numbers into.

    :return p: returns the p-value of the test (alongside the T-value and the defrees of fredom).
    """

    N = len(U)
    expected_num_in_bins = N / num_bins

    # Divied into bins
    counts, something = np.histogram(U, bins=num_bins)

    T = np.sum((counts - expected_num_in_bins) ** 2 / expected_num_in_bins)

    df = num_bins - 1 - 1  # when number of estimated parameters is m=1

    p = 1 - scipy.stats.chi2.cdf(T, df)

    return p, T, df


def kolmogorov_smirnov_unif(data_points: list[float], modulus: int) -> float:
    """
    Implementation of the Kolmogorov-Smirnov test-statistic in O(n * log(n)).
    Tests whether the sampled data_points are uniformly distributed.
    NOTE: could be expanded to be given two CDFs (F_0, F_T) and then just
    find the maximum distance between the two (this fitted our case better).

    :param data_points: the sampled, uniformly distributed data points
    :param modulus: the highest possible value in the sampled data points (parameter for LCG)

    :return: the Kolmogorov-Smirnov test-statistic
    """
    # number of data-points
    num_samples = len(data_points)

    F_0 = np.sort(data_points) / modulus
    F_T = np.linspace(0, 1, num_samples)
    test_statistic = np.abs(F_0 - F_T).max()

    return test_statistic


def up_and_down_test(random_nums: list[float]):
    """
    The up and down test is a test for randomness. It counts the number of runs in a sequence of random numbers.
    A run is defined as a sequence of consecutive numbers that are either increasing or decreasing.
    (See slide 20 on 'slide2bm1.pdf' for more info).

    :param random_nums: a list of random numbers.

    :return Z: the test statistic (I think).
    """
    R = []

    n = len(random_nums)

    previous = None
    run_length = 0
    for i in range(len(random_nums) - 1):
        if random_nums[i] < random_nums[i + 1]:
            if previous == "up" or previous is None:
                run_length += 1
                previous = "up"
            elif previous == "down":
                R.append(run_length)
                run_length = 1
                previous = "up"
        else:
            if previous == "down" or previous is None:
                run_length += 1
                previous = "down"
            elif previous == "up":
                R.append(run_length)
                run_length = 1
                previous = "down"

    count = collections.Counter(R)

    for key, val in count.items():
        expected_val = 0
        if key == 1:
            expected_val = (n + 1) / 12
        elif key == 2:
            expected_val = (11 * n - 4) / 12
        else:
            expected_val = (2 * ((key**2 + 3 * key + 1) * n - (key**3 + 3 * key**2 - key - 4))) / np.math.factorial(key + 3)

        print(f"Expected number of runs of length: {key} is: {np.round(expected_val, 2):>10}")
        print(f"The actual length is: {np.round(val, 2):>26}.00\n")  # <-- this is some scummy code and won't exrapolate to all cases.

    X = len(R)
    Z = (X - (2 * n - 1) / 3) / np.sqrt(((16 * n - 29) / 90))

    return Z


def above_below(random_nums: list[float], median: float) -> tuple[list[int], list[int], int, int, float, float]:
    """
    Calculates the runs for the Above/Below test, as given in the first lecture.
    (See slide 17, on 'slide2bm1.pdf').
    NOTE: This is made for for a sample of uniformly distributed random numbers.
    (So this distribution should have a theoretical median).
    NOTE: The test statistic for this test is given as R_a + R_b.

    :param random_nums: a list of random numbers.
    :param median: the expected median of the random numbers.

    :return R_a: a list of runs of numbers above the median.
    :return R_b: a list of runs of numbers below the median.
    :return n_1: the total number of numbers above the median.
    :return n_2: the total number of numbers below the median.
    :return mu: the expected value of the test statistic.
    :return sigma: the standard deviation of the test statistic.
    """

    R_a = []
    R_b = []

    previous = None
    run_length = 0
    for random_num in random_nums:
        if random_num < median:
            if previous == "below" or previous is None:
                run_length += 1
                previous = "below"
            elif previous == "above":
                R_b.append(run_length)
                run_length = 1
                previous = "below"
        elif random_num > median:
            if previous == "above" or previous is None:
                run_length += 1
                previous = "above"
            elif previous == "below":
                R_a.append(run_length)
                run_length = 1
                previous = "above"

    n_1 = sum(R_a)
    n_2 = sum(R_b)

    mu = 2 * (n_1 * n_2) / (n_1 + n_2) + 1
    sigma = 2 * (n_1 * n_2 * (2 * n_1 * n_2 - n_1 - n_2)) / ((n_1 + n_2) ** 2 + (n_1 + n_2 - 1))

    return R_a, R_b, n_1, n_2, mu, sigma


def up_down(random_nums: list[float]) -> float:
    """
    Knuth's Up/Down test, as given in the first lecture.
    (See slide 18, on 'slide2bm1.pdf').

    :param random_nums: a list of random numbers between 0 and 1.

    :return Z: a list (of something I don't know yet).
    """

    n = len(random_nums)

    # This is a little confusing and maybe not the best code. We index R, such that index
    # 0 corresponds to a run of length 1, index 1 corresponds to a run of length 2, etc.
    R = np.zeros(6)
    run_length = 1
    for i in range(len(random_nums) - 1):
        if random_nums[i] < random_nums[i + 1]:
            run_length += 1
        else:
            run_length = min(run_length, 6)
            R[run_length - 1] += 1

            run_length = 1

    R = R.reshape(-1, 1)  # <-- importent because otherwise (R - n * B) gives a matrix (and we don't want that, no, no).

    A = np.array(
        [
            [4529.4, 9044.9, 13568, 18091, 22615, 27892],
            [9044.9, 18097, 27139, 36187, 45234, 55789],
            [13568, 27139, 40721, 54281, 67852, 83685],
            [18091, 36187, 54281, 72414, 90470, 111580],
            [22615, 45234, 67852, 90470, 113262, 139476],
            [27892, 55789, 83685, 111580, 139476, 172860],
        ]
    )

    B = np.array([[1 / 6], [5 / 24], [11 / 120], [19 / 720], [29 / 5040], [1 / 840]])

    Z = (1 / (n - 6)) * (R - n * B).T @ A @ (R - n * B)

    return Z.item()


#### The actual functions
def plot_hist_and_corr(
    multiplier: int = None,
    shift: int = None,
    modulus: int = None,
    x0: int = None,
    U: list[float] = None,
    break_point: int = None,
    modulus_overwrite: bool = False,
    num_bins: int = None,
    num_scatter_points: int = None,
) -> None:
    """
    This function plots the histogram and corresponding correlation plot of a specific run of our LCG.

    :param multiplier: The multiplier (a) used in the formula.
    :param shift: The shift (c) used in the formula.
    :param modulus: The modulus (M) used in the formula.
    :param x0: initial value (x0) used in the formula. If not provided, a random value will be used.
    :param U: A list of random numbers. If provided, the LCG will not be used.
    :param break_point: The number of iterations to run the generator for.
    :param modulus_overwrite: If you specifically want to overwrite the modulus, set this to True.
    :param num_bins: number of bins in the histogram.
    :param num_scatter_points: number of points plotted in the correlation plot.

    :return: returns void.
    """
    assert U is not None or (
        multiplier is not None and shift is not None and modulus is not None
    ), "You must either provide U or the parameters for the LCG."

    if num_bins is None:
        num_bins = 10

    if x0 is None:
        # This is dumb, but we roll
        x0 = np.random.randint(0, modulus)

    if num_scatter_points is None or num_scatter_points > (modulus // 10):
        num_scatter_points = min(100, modulus // 10)

    if U is None:
        # In this case, we need to generate U
        _, U = linear_congruental_generator(
            multiplier=multiplier, shift=shift, modulus=modulus, x0=x0, break_point=break_point, modulus_overwrite=modulus_overwrite
        )
        hist_title = f"Histogram of random numbers, with: \nM: {modulus}, a: {multiplier}, c: {shift}, x0: {x0}"

    elif multiplier is not None and shift is not None and modulus is not None and x0 is not None:
        # In this case we already generated U, but it was generated with LCG
        hist_title = f"Histogram of random numbers, with: \nM: {modulus}, a: {multiplier}, c: {shift}, x0: {x0}"
    else:
        # In this case we have generated U, whcih we don't know how was generated
        hist_title = f"Histogram of {len(U)} random numbers (not generated by LCG))"

    # counts, bins = np.histogram(U_hist, bins=num_bins)
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.title(hist_title)
    plt.hist(U, bins=num_bins)

    plt.subplot(1, 2, 2)
    plt.title("Correlation plot of $U_{i - 1}$ and $U_i$")
    plt.scatter(U[:num_scatter_points], U[1 : (num_scatter_points + 1)], s=[0.1] * num_scatter_points)

    plt.show()


def chungus_test(
    multiplier: int = None,
    shift: int = None,
    modulus: int = None,
    x0: int = None,
    U: list[float] = None,
    break_point: int = None,
    modulus_overwrite: bool = False,
    significance_level: float = 0.05,
    chi_squared_bins: int = 10,
    expected_median: float = 0.5,
    plot: bool = False,
    num_bins: int = None,
    num_scatter_points: int = None,
) -> None:
    """
    BIG test! Performs the following tests:
    Distribution tests:
        - Chi-squared test
        - Kolmogorov-Smirnov test
    Correlation tests:
        - Runs above/below median test
        - Knuth's up/down test
        - Up and down test

    :param multiplier: the multiplier (a) in the linear congruental generator.
    :param shift: the shift (c) in the linear congruental generator.
    :param modulus: the modulus (M) in the linear congruental generator.
    :param x0: the initial value (x0) in the linear congruental generator.
    :param U: A list of random numbers. If provided, the LCG will not be used.
    :param break_point: the break point (b) in the linear congruental generator.
    :param modulus_overwrite: If you specifically want to overwrite the modulus, set this to True.
    :param significance_level: the significance level for the tests.
    :param chi_squared_bins: the number of bins to use in the chi-squared test.
    :param expected_median: the expected median of the random numbers.
    :param plot: whether to plot the random numbers.
    :param num_bins: the number of bins to use in the histogram.
    :param num_scatter_points: the number of scatter points to use in the scatter plot.
    """
    assert U is not None or (
        multiplier is not None and shift is not None and modulus is not None and x0 is not None
    ), "You must either provide U or the parameters for the LCG."

    if U is None:
        _, U = linear_congruental_generator(
            multiplier=multiplier, shift=shift, modulus=modulus, x0=x0, break_point=break_point, modulus_overwrite=modulus_overwrite
        )
        print(f"{len(U)} random numbers generated using the Linear Congruental Generator with parameters:")
        print(f"Multiplier (a): {multiplier}")
        print(f"shift (c): {shift}")
        print(f"modulus (M): {modulus}")
        print(f"Initial value (x0): {x0}")

    else:
        modulus = len(U)
        print(f"{modulus} random numbers provided.")
        print(f"Will now test for uniformity.")

    print("\nMultiple tests were performed: \n")

    print(">>>> FIRSTLY from the distribution tests <<<<")
    print(f"From the Chi-squared test for {chi_squared_bins} bins:")
    p_chi_sqr, T_chi_sqr, df_chi_sqr = chi_squared_test_U(U, chi_squared_bins)
    print(f"The test statistic T = {T_chi_sqr} \nshould be asymptotically chi-squared with {df_chi_sqr} degrees of freedom.")
    print(f"We test that by p-value. The p-value is: {p_chi_sqr}.")
    print(
        f"With a significance level of: {significance_level} we find that the test is {'not ' if p_chi_sqr < significance_level else ''}significant.\n"
    )

    print("From the Kolmogorov-Smirnov test:")
    D_n = kolmogorov_smirnov_unif(U, modulus)
    p_KS = 1 - scipy.special.kolmogorov(D_n)
    print(f"The test-statistic: {D_n}, \nshould be asymptotically K-S distributed.")
    print(f"We test that by p-value. The p-value is: {p_KS}")
    print(
        f"With a significance level of: {significance_level} we find that the test is {'not ' if p_KS < significance_level else ''}significant.\n"
    )

    print("\n>>>> SECONDLY from the correlation tests <<<<")
    print("From the Up-and-Down test we find: ")
    Z_up_and_down = up_and_down_test(U)
    p_up_and_down = 2 * (1 - scipy.stats.norm.cdf(abs(Z_up_and_down), loc=0, scale=1))

    print(f"This test statistic Z = {Z_up_and_down} \nshould be asymptotically N(0, 1).")
    print(f"We test that by p-value. The p-value is: {p_up_and_down}.")
    print(
        f"With a significance level of: {significance_level} we find that the test is {'not ' if p_up_and_down < significance_level else ''}significant.\n"
    )

    print("From the Above/ Below test we find: ")
    R_a, R_b, n_1, n_2, mu, sigma = above_below(
        U, expected_median
    )  # <-- n_1 and n_2 are the number of observations above and below the expected median (here unused).
    T_above_below = len(R_a) + len(R_b)
    p_above_below = 2 * (1 - scipy.stats.norm.cdf(abs(T_above_below), loc=mu, scale=sigma))

    print(f"This test statistic T = {T_above_below} \nshould be asymptotically N({round(mu, 2)}, {round(sigma, 2)}).")
    print(f"We test that by p-value. The p-value is: {p_above_below}.")
    print(
        f"With a significance level of: {significance_level} we find that the test is {'not ' if p_above_below < significance_level else ''}significant.\n"
    )

    print("From the Up/ Down test we find: ")
    Z_up_down = up_down(U)
    p_up_down = 1 - scipy.stats.chi2.cdf(x=Z_up_down, df=6)
    print(f"The Z from the up-down test: {Z_up_down}. \nShould be compared with a chi^2(6) dsitribution.")
    print(f"We test that by p-value. The p-value is: {p_up_down}.")
    print(
        f"With a significance level of: {significance_level} we find that the test is {'not ' if p_up_down < significance_level else ''}significant.\n\n"
    )

    if plot:
        print("\nFinally a histogram and correlation plot is given here: \n")
        plot_hist_and_corr(
            multiplier=multiplier, shift=shift, modulus=modulus, x0=x0, U=U, num_bins=num_bins, num_scatter_points=num_scatter_points
        )


### >>>> MCMC (ex06) <<<< ###
def normpdf(x: float, mu: float, sigma: float) -> float:
    """
    Normal distribution probability density function.

    :param x: x value
    :param mu: mean
    :param sigma: standard deviation

    :return: probability density function value
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1.0 / 2 * ((mu - x) / sigma) ** 2)


def MCMC_1(num_samples: int, g: Callable[[float], float], X_0: int = None, burn_in_ratio: float = 0.5):
    """
    Markov Chain Monte Carlo scheme. We use the Metropolis-Hastings algorithm.
    NOTE: The burn_in_ratio is added to the amount samples. I.e. the number of iterations
    are num_samples * (1 + burn_in_ratio).

    :param num_samples: Number of samples points from the MCMC
    :param X_0: Initial value of the Markov Chain. If None, a random value between 0 and m is chosen.
    :param burn_in_ratio: The amount of burn-in we wish to apply (as a ratio)

    :return: Samples from the Markov Chain
    """
    num_iter = int(num_samples * (1 + burn_in_ratio))
    if X_0 is None:
        X_0 = np.random.randint(0, m + 1)  # Random number between 0 and m (inclusive)

    samples = []
    samples.append(X_0)

    for _ in range(num_iter):
        delta_X = np.random.binomial(n=1, p=0.5) * 2 - 1  # Number either -1 or 1
        X_curr = samples[-1]

        X_proposal = (
            X_curr + delta_X
        )  # <-- okay.. this is technically unecessary, but it's good practice to initialize variables (that would otherwise go out of scope)

        if X_proposal < 0:
            # Edge-case lower end
            X_proposal = m
        elif X_proposal > m:
            # Edge-case upper end
            X_proposal = 0

        A = min(1.0, g(X_proposal) / g(X_curr))

        accept = np.random.binomial(n=1, p=A)

        if accept:
            samples.append(X_proposal)
        elif not accept:
            samples.append(X_curr)

    return samples[:num_samples]


def MCMC_two_dim_boxed(num_samples: int, g: Callable[[float], float], X_0: int = None, burn_in_ratio: float = 0.5, seed: int = 42):
    """
    Markov Chain Monte Carlo scheme. We use the Metropolis-Hastings algorithm.
    This is for the two-dimensional case.
    NOTE: The burn_in_ratio is added to the amount samples. I.e. the number of iterations
    are num_samples * (1 + burn_in_ratio).

    :param num_samples: Number of samples points from the MCMC
    :param X_0: Initial value of the Markov Chain. If None, a random value between 0 and m is chosen.
    :param burn_in_ratio: The amount of burn-in we wish to apply (as a ratio)

    :return: Samples from the Markov Chain
    """

    # initialization
    np.random.seed(seed=seed)
    num_iter = int(num_samples * (1 + burn_in_ratio))

    # Finidng our first guess (if none specified)
    if X_0 is None:
        first_sample = np.random.randint(0, m + 1)
        X_0 = np.array(
            [first_sample, np.random.randint(0, 10 - first_sample + 1)]
        )  # [Random number between 0 and m (inclusive), Random number between 0 and 10 - previously sampled number (inclusive)]

    samples = [X_0]

    for _ in range(num_iter):
        X_curr = samples[-1]

        # random guess
        delta_X = np.random.randint(-1, 2, size=2)

        # We can only move so discard a random walk if we are not moving
        while delta_X[0] == 0 and delta_X[1] == 0:
            delta_X = np.random.randint(-1, 2, size=2)

        X_proposal = X_curr + delta_X

        # if out of bounds we just stay at the same point
        if X_proposal.sum() > m or X_proposal[0] < 0 or X_proposal[1] < 0:
            samples.append(X_curr)

        else:
            A = min(1, g(X_proposal) / g(X_curr))
            accept = np.random.binomial(n=1, p=A)

            if accept:
                samples.append(X_proposal)
            elif not accept:
                samples.append(X_curr)

    return samples[:num_samples]


### >>>> Blocking system (ex04) <<<< ###
# Define blocking system
m = 10  # service units
SERVICE_MEAN = 8  # 1/mean service time
ARRIVAL_MEAN = 1  # mean arrival time


HYPER_PROB = 0.8
HYPER_MEAN = [1 / 0.8333, 1 / 5.0]


class Event:
    def __init__(
        self,
        event_type: str,
        time: float,
    ) -> None:
        if event_type.lower() not in ["arrival", "departure"]:
            raise ValueError("event_type must be either arrival or departure")
        self.event_type = event_type
        self.time = time


def sample_arrival_poisson_process() -> float:
    """
    Sample the arrival time from a Poisson process.

    :return: arrival time (as a float)
    """
    return np.random.exponential(ARRIVAL_MEAN)  # TODO: check if is it not 1/arrial_mean


def sample_arrival_hyper_exponential() -> float:
    """ """
    p = np.random.binomial(n=1, p=HYPER_PROB)

    if p:
        return np.random.exponential(HYPER_MEAN[0])
    else:
        return np.random.exponential(HYPER_MEAN[1])


def sample_service_time_exponential() -> float:
    """
    Sample the service time from an exponential distribution.

    :return: service time (as a float)
    """
    return np.random.exponential(SERVICE_MEAN)


def check_available_service(service_units: list[bool]) -> tuple[int, bool]:
    """ """
    for indx, unit_occupied in enumerate(service_units):
        if not unit_occupied:
            return indx, True

    return None, False


def apend_event(event_list: list[Event], event_to_append: Event) -> list[Event]:
    """ """
    for indx, event in enumerate(event_list):
        if event.time > event_to_append.time:
            event_list.insert(indx, event_to_append)
            return event_list

    event_list.append(event_to_append)
    return event_list


## Simulation
def blocking_simulation(
    simulation_runs: int = 10,
    m: int = 10,
    N: int = 10000,
    sample_arrival: Callable[[], float] = sample_arrival_poisson_process,
    sample_service_time: Callable[[], float] = sample_service_time_exponential,
) -> tuple[list[float], list[float]]:
    """
    A function for runinng multiple simulations of a blocking system.

    :param simulation_runs: number of simulations to run
    :param m: number of service units
    :param N: number of customers
    :param sample_arrival: function for sampling arrival time
    :param sample_service_time: function for sampling service time

    :return: list of blocked fractions and list of average arrival times
    """
    # NOTE: Maybe use burn in period...?

    blocked_fractions = []
    arrival_times = []
    for i in range(simulation_runs):
        print(f"run {i+1}")
        custmer_count = 0
        global_time = 0
        event_counter = 0
        block_count = 0
        arrivals = 0

        # lists
        event_list = []
        service_units_status = [False for _ in range(m)]  # <-- Indicates whether the service units are occupied or not

        # First arrival
        first_arrival = sample_arrival()
        event_list.append(Event("arrival", global_time + first_arrival))
        arrivals += first_arrival

        global_time += first_arrival
        event_list.append(Event("departure", global_time + sample_service_time()))
        service_units_status[0] = True  # <-- unit 1 is occupied

        while custmer_count < N:
            current_event = event_list[event_counter]

            # Increment global time
            global_time = current_event.time

            if current_event.event_type == "arrival":
                custmer_count += 1

                # Check for free service units
                indx, available = check_available_service(service_units_status)

                if available:
                    # Insert departure event and depend to eventlist

                    departure_event = Event("departure", global_time + sample_service_time())
                    event_list = apend_event(event_list, departure_event)

                    # Take service unit
                    service_units_status[indx] = True  # <-- unit indx is occupied

                if not available:
                    # Costumer blocked
                    block_count += 1

                # insert time for next arrival
                new_arrival = sample_arrival()  # arrival[customer_count]
                arrivals += new_arrival
                arrival_event = Event("arrival", global_time + new_arrival)
                event_list = apend_event(event_list, arrival_event)

            elif current_event.event_type == "departure":
                # Free the service unit for the current departure event
                for indx, unit_occupied in enumerate(service_units_status):
                    if unit_occupied:
                        service_units_status[indx] = False  # <-- unit indx is free
                        break

            # increment event counter
            event_counter += 1

        blocked_fractions.append(block_count / N)
        arrival_times.append(arrivals / N)

    return blocked_fractions, arrival_times
