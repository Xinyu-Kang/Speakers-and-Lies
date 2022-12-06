from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import scipy

dataDir = "<replace with data directory path>"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        Sigma_m = self.Sigma[m]
        d = len(Sigma_m)
        mu_m = self.mu[m]
        inside_sum = (mu_m * mu_m) / (2 * Sigma_m) + 0.5 * np.log(Sigma_m)
        result = - np.sum(inside_sum) - 0.5 * d * np.log(2 * np.pi)
        return result

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    Sigma_m = myTheta.Sigma[m][np.newaxis, :]
    mu_m = myTheta.mu[m][np.newaxis, :]
    if len(x.shape) == 2:
        T = x.shape[0]
        Sigma_m = np.repeat(Sigma_m, T, axis=0)
        mu_m = np.repeat(mu_m, T, axis=0)
    first_term = (0.5 * (x * x) / Sigma_m) - (mu_m * x / Sigma_m)
    first_term = - np.sum(first_term, axis=1)
    second_term = myTheta.precomputedForM(m)
    result = first_term + second_term
    if len(x.shape) == 1:
        result = float(result)
    return result


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    nominator = np.log(myTheta.omega) + log_Bs
    dominator = scipy.special.logsumexp(log_Bs, axis=0, b=myTheta.omega)
    return nominator - dominator


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_p = scipy.special.logsumexp(log_Bs, axis=0, b=myTheta.omega)
    L = np.sum(log_p)
    return L


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker.
    Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    init_omega = np.ones_like(myTheta.omega) * (1 / M)
    init_mu = X[np.random.choice(X.shape[0], M, replace=False), :]
    init_Sigma = np.ones_like(myTheta.Sigma)
    myTheta.reset_omega(init_omega)
    myTheta.reset_mu(init_mu)
    myTheta.reset_Sigma(init_Sigma)

    i = 0
    prev_L = float("-inf")
    T, d = X.shape
    while i < maxIter:
        log_Bs = np.zeros((M, T))
        for m in range(M):
            b = log_b_m_x(m, X, myTheta)
            log_Bs[m] = b
        p = np.exp(log_p_m_x(log_Bs, myTheta))
        L = logLik(log_Bs, myTheta)
        sum_p = np.sum(p, axis=1)
        myTheta.reset_omega(sum_p / T)
        myTheta.reset_mu(np.dot(p, X) / sum_p[:, np.newaxis])
        myTheta.reset_Sigma(np.dot(p, X * X) / sum_p[:, np.newaxis] - myTheta.mu * myTheta.mu)
        improvement = L - prev_L
        if improvement < epsilon:
            break
        prev_L = L
        i += 1
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    correctModel = models[correctID].name
    T, d = mfcc.shape
    models_logLik = []
    for theta in models:
        M = theta._M
        log_Bs = np.zeros((M, T))
        for m in range(M):
            b = log_b_m_x(m, mfcc, theta)
            log_Bs[m] = b
        L = logLik(log_Bs, theta)
        models_logLik.append((L, theta.name))
    models_logLik.sort(reverse=True)
    if len(models_logLik) > 0:
        bestModel = models_logLik[0][1]
    fout = open("gmmLiks.txt", 'a')
    fout.write("\n{}\n".format(correctModel))
    for i in range(k):
        if i < len(models_logLik):
            fout.write("{} {}\n".format(models_logLik[i][1], models_logLik[i][0]))
        else:
            break
    return 1 if (bestModel == correctModel) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []

    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0
    maxIter = 20
    S = 32
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        i = 1
        for speaker in dirs:
            if i > S:
                break

            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            i += 1

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)

    print("numCorrect: {}".format(numCorrect))
    print("accuracy: {}".format(accuracy))
