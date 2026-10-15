import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import os
from scipy.stats import norm
from scipy.sparse.linalg import LinearOperator, eigs

######################################################################
# define general parameters

np.random.seed(42)              # fixed random seed for reproducable results
T = 1                           # time interval [0,T] for all computations
nt = 2000                       # time resolution
dt = T / nt                     # uniform time step dt
t = np.linspace(0., T, nt + 1)  # time array
z = 3.                          # observable value for which P(f(X_T^eps) > z) is evaluated
eps = 0.5                       # noise strength for direct sampling and importance sampling

######################################################################
# define example-specific functions

dim = 2                         # spatial dimension n of the model
getB          = lambda x        : np.array([-x[0] * x[1], x[0]**2])                       # drift vector field b (without linear terms)
getGradB      = lambda x, dx    : np.array([[-x[1], -x[0]], [2. * x[0], 0. * x[0]]]) @ dx # Jacobian of b times dx
getGradBT     = lambda x, dx    : np.array([[-x[1], 2. * x[0]], [-x[0], 0. * x[0]]]) @ dx # transposed Jacobian of b times dx
getHessBTheta = lambda x, p, dx : np.array([[2. * p[1], -p[0]], [-p[0], 0. * x[0]]]) @ dx # \partial_i \partial_j b_k(x) p_k dx_j
getIF         = lambda x, dt    : np.array([x[0] * np.exp(-dt), x[1] * np.exp(-4. * dt)]) # integrating factor for linear terms in b times x
getIFT        = lambda x, dt    : getIF(x, dt)                                            # transposed integrating factor for linear terms in b times x
getIFRic      = lambda x, dt    : np.array([[x[0,0] * np.exp(-2. * dt), x[0,1] * np.exp(-5. * dt)],[x[1,0] * np.exp(-5. * dt), x[1,1] * np.exp(-8. * dt)]]) # integrating factor for Riccati eq.
getF          = lambda x        : x[0] + 2. * x[1]              # observable
getGradF      = lambda x, dx    : dx[0] + 2. * dx[1]            # observable gradient times dx
getGradFT     = lambda x, dx    : dx * np.array([1., 2.])       # just the observable gradient vector
getHessF      = lambda x, dx    : np.zeros_like(dx)             # Hessian matrix of observable
getSigma      = lambda x        : np.array([x[0], 0.5 * x[1]])  # forcing correlation in SDE
getA          = lambda x        : np.array([x[0], 0.25 * x[1]]) # diffusion matrix a = sigma sigma^\top
getAInverse   = lambda x        : np.array([x[0], 4. * x[1]])   # inverse of diffusion matrix

######################################################################
# function for direct sampling to get tail probability and sample paths

def getSamplePaths(nPaths = 100, nParallel = int(1e4), conf = 0.95):
    print('Performing direct sampling for paths with f(X_T^eps) > z with z = {} and eps = {}'.format(z, eps))
    print('Aiming for {} samples, with {} simluations in parallel'.format(nPaths, nParallel))
    i = 0
    nHits = 0
    retPaths = np.zeros((nt + 1, dim, nPaths))
    x = np.zeros((nt + 1, dim, nParallel))
    while nHits < nPaths:
        i += 1
        x[0] *= 0.
        for j in range(nt):
            x[j + 1] = getIF(x[j] + dt * getB(x[j]) + np.sqrt(eps * dt) * getSigma( \
                      np.random.randn(dim, nParallel)), dt)
        currentHits = np.where(getF(x[-1]) > z)[0]
        retPaths[:, :, nHits : min(nPaths, nHits + len(currentHits))] = x[:, :, currentHits[:min(len(currentHits), nPaths - nHits)]]
        nHits += len(currentHits)
        sys.stdout.write("\rSimulation no. %d with %d hits" % (i, nHits))
        sys.stdout.flush()
    nSamples = i * nParallel
    prob = nHits / nSamples
    std  = np.sqrt(prob * (1. - prob) /nSamples)
    c = norm.ppf((conf + 1) / 2.)
    print('')
    print('Total number of samples computed: {}'.format(nSamples))
    print('Estimated tail probability from this: {}'.format(prob))
    print('Asymptotic {}% confidence interval: [{},{}]'.format( \
            100 * conf, round(prob - c * std, int(-np.log10(prob) + 3)), \
            round(prob + c * std, int(-np.log10(prob) + 3))))
    return retPaths, prob, std
    
######################################################################

def getTimeIntegral(a, b):
    ret = np.sum(a * b, axis = 1) * dt
    return np.sum(ret[:-1])
    
######################################################################
# instanton
gradThresh = 1e-6               # when to stop gradient descent for instanton
descentStepSizeInitial = 1.     # start value for armijo line search
descentStepSizeMinimum = 1.e-7  # end, break if reached
armijoConst = 1.e-2             # constant for armijo sufficient decrase condition
descentStepSizeDecrease = 0.5   # decrease trial step size by this factor

class Instanton():
    def __init__(self):
        self.phi = np.zeros((nt + 1, dim))
        self.theta = np.zeros((nt + 1, dim))
        self.thetahat = np.zeros((nt + 1, dim))
        
    def getGradientFromPandZ(self):
        return getA((self.theta - self.thetahat).T).T
        
    def integratePhi(self):
        self.phi[0] *= 0.
        for i in range(nt):
            self.phi[i + 1] = getIF(self.phi[i] + dt * (getB(self.phi[i]) + getA(self.theta[i])), dt)
    
    def integrateThetaHat(self):
        for i in range(2, nt + 1):
            self.thetahat[nt - i] = getIFT(self.thetahat[nt + 1 - i] + dt * getGradBT(self.phi[nt + 1 - i], self.thetahat[nt + 1 - i]), dt)

    def integrateForwardAndBackward(self, lbda, mu, targetObservable):
        self.integratePhi()
        obs = getF(self.phi[-1])
        self.thetahat[-2] = getIFT(getGradFT(self.phi[-1], lbda + mu * (targetObservable - obs)), dt)
        self.integrateThetaHat()
        return obs
    
    def armijoLineSearch(self, gradient, s, obsPrevious, lbda, mu, targetObservable):
        obs    = obsPrevious
        sigma  = descentStepSizeInitial
        thetachis  = getTimeIntegral(self.theta, getA(s.T).T)
        schis  = getTimeIntegral(s, getA(s.T).T)
        gradSs = getTimeIntegral(gradient, s)
        print('Action = ', 0.5 * getTimeIntegral(self.theta, getA(self.theta.T).T))
        while sigma > descentStepSizeMinimum:
            self.theta = self.theta + sigma * s
            self.integratePhi()
            obsNew = getF(self.phi[-1])
            breakConditionRHS = - armijoConst * sigma * gradSs
            breakConditionRHS = breakConditionRHS + lbda * (obsNew - obsPrevious) + mu / 2. * ((targetObservable - obsPrevious)**2 - (targetObservable - obsNew)**2)
            breakCondition    = (sigma * thetachis + sigma**2 / 2. * schis <= breakConditionRHS)
            if breakCondition:
                obs = obsNew
                break
            else:
                self.theta = self.theta - sigma * s
                sigma = sigma * descentStepSizeDecrease
        return sigma, obs
        
    def gradientDescent(self, lbda, targetObservable = [1.], mu = 0., initialTheta = None):
        print('################################################')
        print("Computing Instanton for fixed penalty parameter mu = {}".format(mu))
        print("and lagrange multiplier lbda = {} and target observable = {}".format(lbda, targetObservable))
        print("Performing updates p^{k+1} = p^k - alpha (p^k - z^k)")
        print("where alpha is found via Armijo line search,")
        print("Terminates if gradient L^2 norm is below eps.")
        print("Corresponds to gradient descent (preconditioned with chi^{-1})")
        print("################################################")
        # initialize fields
        if initialTheta is None:
            self.theta = np.random.randn(nt + 1, dim)
        else:
            self.theta = initialTheta
        self.phi *= 0.
        self.thetahat *= 0.
        # define counters and iteration parameters
        obsValue = 0.
        gradientNorm = np.finfo(np.float32).max
        iterationCounter = 0
        errorTol = copy.copy(gradThresh)
        while((gradientNorm > errorTol or iterationCounter < 2) and iterationCounter < 5e3):
            # compute gradient, current observable value and gradient norm
            obsPrevious = self.integrateForwardAndBackward(lbda, mu, targetObservable)
            obsValue = obsPrevious
            gradient = self.getGradientFromPandZ()
            s = -gradient
            gradientNorm = np.sqrt(getTimeIntegral(gradient, gradient))
            print('Norm of gradient =', gradientNorm)
            if iterationCounter == 0:
                errorTol = max(gradientNorm * errorTol, 1e-6)
                print('Trying to reach a gradient norm of', errorTol)
            s = getAInverse(s.T).T
            # armijo line search
            sigma, obsValue = self.armijoLineSearch(gradient, s, obsPrevious, lbda, mu, targetObservable)
            if sigma <= descentStepSizeMinimum:
                print('Stopping the iteration, no valid step size in gradient direction')
                break
            iterationCounter += 1
            # some final output
            print('Iteration =', iterationCounter)
            print('observableValue =', obsValue)
            print('stepSize =', sigma)
            print('######')
        action = 0.5 * getTimeIntegral(self.theta, getA(self.theta.T).T)
        obsValue = self.integrateForwardAndBackward(lbda, mu, targetObservable)
        gradient = self.getGradientFromPandZ()
        gradientNorm = np.sqrt(getTimeIntegral(gradient, gradient))
        print('################################################')
        print('Parameters of the solution:')
        print('Iteration', iterationCounter)
        print('Gradient norm =', gradientNorm)
        print('lambda =', lbda)
        print('mu =', mu)
        print('observable =', obsValue)
        print('Action =', action)
        ret = obsValue, action, lbda, copy.copy(self.theta), copy.copy(self.phi)
        dS = 0.5 * np.real(np.sum(self.theta * getA(self.theta.T).T, axis = 1) * dt)
        ret = ret + (dS,)
        return ret
        
    def searchInstantonViaAugmented(self, targetObservable, muMin = np.log10(5.), muMax = np.log10(50.), nMu = 5, initLbda = 1.):
        print("Find instanton for observable value", targetObservable, "via augmented Lagrangian method.")
        muList = np.logspace(muMin, muMax, nMu)
        obsValue, action, lbda, theta, phi, dS = self.gradientDescent(initLbda, targetObservable = targetObservable, mu =  muList[0])
        print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[0], lbda, obsValue, action))
        lbda = muList[0] * (targetObservable - obsValue) + initLbda
        for j in range(1, nMu):
            obsValue, action, lbda, theta, phi, dS = self.gradientDescent(lbda, targetObservable = targetObservable, mu =  muList[j], initialTheta = theta)
            print("Mu = {}, lambda = {} yields observable = {} and action = {}".format(muList[j], lbda, obsValue, action))
            lbda = muList[j] * (targetObservable - obsValue) + lbda
        return obsValue, action, lbda, theta, phi, dS
        
######################################################################
# importance sampling for better transition path statistics

def getTransitionPathStatisticsImportanceSampling(phi, theta, lbda, nPaths = int(1e5), \
        nParallel = int(1e4), delta = 0.05, \
        times = [0.05, 0.25, 0.5, 0.75, 0.95], nbins = 200, xmin = -1.2, xmax = 2.8, ymin = -0.6, ymax = 1.3):
    print('Performing importance sampling for paths with f(X_T^eps) [z - delta, z + delta]')
    print('with z = {}, delta = {} and eps = {}'.format(z, delta, eps))
    print('Aiming for {} samples, with {} simluations in parallel'.format(nPaths, nParallel))
    i = 0
    nHits = 0
    samplesTimes = np.zeros((len(times), dim, nPaths))
    reweight = np.zeros(nPaths)
    meanPath = np.zeros((nt + 1, dim))
    if dim == 2:
        fullHist = np.zeros((nbins, nbins))
        xbins = np.linspace(xmin, xmax, nbins + 1)
        ybins = np.linspace(ymin, ymax, nbins + 1)
    x = np.zeros((nt + 1, dim, nParallel))
    reweightTmp = np.zeros(nParallel)
    while nHits < nPaths:
        i += 1
        x[0] *= 0.
        reweightTmp *= 0.
        for j in range(nt):
            reweightTmp += dt * np.sum((getB((phi[j])[:, None] + np.sqrt(eps) * x[j]) - getB(phi[j])[:, None] - np.sqrt(eps) * getGradB(phi[j], x[j])) / eps * theta[j][:, None], axis = 0)
            x[j + 1] = getIF(x[j] + dt * (getB((phi[j])[:, None] + np.sqrt(eps) * x[j]) - getB(phi[j])[:, None]) / np.sqrt(eps) + np.sqrt(dt) * getSigma( \
                      np.random.randn(dim, nParallel)), dt)
        reweightTmp += lbda / eps * (getF(phi[-1, :, None] + np.sqrt(eps) * x[-1]) - getF(phi[-1,:,None]) - np.sqrt(eps) * getGradF(phi[-1], x[-1]))
        reweightTmp = np.exp(reweightTmp)
        currentHits = np.where(np.abs(getF(x[-1])) < delta)[0]
        for k in range(len(times)):
            samplesTimes[k, :, nHits : min(nPaths, nHits + len(currentHits))] = x[int(times[k] / dt), :, currentHits[:min(len(currentHits), nPaths - nHits)]].T
        reweight[nHits : min(nPaths, nHits + len(currentHits))] = reweightTmp[currentHits[:min(len(currentHits), nPaths - nHits)]]
        if dim == 2:
            fullHist += np.histogram2d(((phi[:,0])[:, None] + np.sqrt(eps) * x[:, 0, currentHits[:min(len(currentHits), nPaths - nHits)]]).flatten(),\
                                       ((phi[:,1])[:, None] + np.sqrt(eps) * x[:, 1, currentHits[:min(len(currentHits), nPaths - nHits)]]).flatten(),\
                                       weights = np.tile(reweightTmp[currentHits[:min(len(currentHits), nPaths - nHits)]], (nt + 1, 1)).flatten(),\
                                       bins = [xbins, ybins])[0]
        meanPath += np.sum((phi[:,:, None] + np.sqrt(eps) * x[:, :, currentHits[:min(len(currentHits), nPaths - nHits)]]) * reweightTmp[None, None, currentHits[:min(len(currentHits), nPaths - nHits)]], axis = 2)
        nHits += len(currentHits)
        sys.stdout.write("\rSimulation no. %d with %d hits" % (i, nHits))
        sys.stdout.flush()
    meanPath = meanPath / np.sum(reweight)
    nSamples = i * nParallel
    ret = (meanPath, samplesTimes, times, reweight)
    if dim == 2 : ret = ret + (fullHist, xbins, ybins)
    return ret

######################################################################
# Riccati solution

def solveForwardRiccati(phi, theta, lbda):
    Q = np.zeros((nt + 1, dim, dim))
    expint = 0.
    for i in range(nt):
        expint += 0.5 * np.sum(np.trace(getHessBTheta(phi[i], theta[i], Q[i]))) * dt
        Q[i+1] = getIFRic(Q[i] + dt * (getA(np.eye(dim)) + getGradB(phi[i], Q[i]) + getGradB(phi[i], Q[i]).T + Q[i] @ getHessBTheta(phi[i], theta[i], Q[i])), dt)
    U = np.eye(dim) - lbda * getHessF(phi[-1], Q[-1])
    expint = np.exp(expint)
    cf = expint / lbda / np.sqrt(np.linalg.det(U) * np.sum(getGradFT(phi[-1],1.) * (Q[-1] @ np.linalg.inv(U) @ getGradFT(phi[-1],1.))))
    print('C_f from Riccati: {}'.format(cf))
    return cf, Q
    
######################################################################
# second variation operator

class SecondVariationOperator(LinearOperator):
    
    def __init__(self, phi, theta, lbda, useEtaPerpProjection = True):
        self.phi = phi
        self.theta = theta
        self.lbda = lbda
        self.shape  = (dim * (nt + 1), dim * (nt + 1))
        self.dtype = np.dtype('double')
        super().__init__(dtype=self.dtype, shape=self.shape)
        self.useEtaPerpProjection = useEtaPerpProjection
        if self.useEtaPerpProjection:
            self.eta       = getSigma(self.theta.T).T
            self.eta_norm  = np.sqrt(getTimeIntegral(self.eta, self.eta))
        self.counter = 0
            
    def _matvec(self, inp):
        self.counter += 1
        print('Application no. {} of operator'.format(self.counter))
        inpp = np.reshape(inp, (nt + 1, dim))
        if self.useEtaPerpProjection:
            inpp = inpp - getTimeIntegral(self.eta, inpp) * self.eta / self.eta_norm**2 
        self.gamma = np.zeros((nt + 1, dim))
        for i in range(nt - 1):
            self.gamma[i+1] = getIF(self.gamma[i] + dt * (getGradB(self.phi[i], self.gamma[i]) + getSigma(inpp[i])), dt)
        zeta = np.zeros((nt + 1, dim))
        zeta[-2] = getIFT(self.lbda * getHessF(self.phi[-1], self.gamma[-1]), dt)
        for i in range(2, nt + 1):
            zeta[nt - i] = getIFT(zeta[nt + 1 - i] + dt * (getHessBTheta(self.phi[nt + 1 - i], self.theta[nt + 1 - i], self.gamma[nt + 1 - i]) + getGradBT(self.phi[nt + 1 - i], zeta[nt + 1 - i])), dt)
        ret = getSigma(zeta.T).T
        if self.useEtaPerpProjection:
            ret = ret - getTimeIntegral(self.eta, ret) * self.eta / self.eta_norm**2 
        return ret.flatten()
        
######################################################################

if __name__ == '__main__':
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    
    np.save(data_dir + '/target_obs.npy', z)
    ############################################################
    # direct sampling for paths that exceed observable threshold,
    # this might take some time
    
    paths, prob, std = getSamplePaths()
    np.save(data_dir + '/direct_sampling_paths_eps_{}.npy'.format(eps), paths)
    np.save(data_dir + '/direct_sampling_prob_eps_{}.npy'.format(eps), prob)
    np.save(data_dir + '/direct_sampling_prob_std_eps_{}.npy'.format(eps), std)
    
    ############################################################
    # instanton computation
    
    instanton = Instanton()
    obsValue, action, lbda, theta, phi, dS = instanton.searchInstantonViaAugmented(z)
    np.save(data_dir + '/inst_obs.npy', obsValue)
    np.save(data_dir + '/inst_act.npy', action)
    np.save(data_dir + '/inst_lbda.npy', lbda)
    np.save(data_dir + '/inst_theta.npy', theta)
    np.save(data_dir + '/inst_phi.npy', phi)
    np.save(data_dir + '/inst_ds.npy', dS)
    ############################################################
    # solve Riccati eq. along instanton, needs instanton to be computed first
    
    phi = np.load(data_dir + '/inst_phi.npy')
    theta = np.load(data_dir + '/inst_theta.npy')
    lbda = np.load(data_dir + '/inst_lbda.npy')
    cf, Q = solveForwardRiccati(phi, theta, lbda)
    np.save(data_dir + '/ric_cf.npy', cf)
    np.save(data_dir + '/ric_Q.npy', Q)
    ############################################################
    # find eigenvalues of A_z, needs instanton to be computed first
    
    phi = np.load(data_dir + '/inst_phi.npy')
    theta = np.load(data_dir + '/inst_theta.npy')
    lbda = np.load(data_dir + '/inst_lbda.npy')
    action = np.load(data_dir + '/inst_act.npy')
    A = SecondVariationOperator(phi, theta, lbda)
    evals, evecs = eigs(A, 200, which = 'LM', tol = 1E-8)
    cf = 1. / np.sqrt(2. * action * np.prod(1. - evals))
    print('C_f from eigenvalues: {}'.format(cf))
    evecs = evecs / np.sqrt(np.sum(evecs**2 * dt, axis = 0))[None, :]
    np.save(data_dir + '/evals.npy', evals)
    np.save(data_dir + '/evecs.npy', evecs)
    np.save(data_dir + '/evals_cf.npy', cf)
    gamma = np.zeros((nt + 1, dim, len(evals)))
    for i in range(len(evals)):
        A(evecs[:, i])
        gamma[:,:,i] = copy.copy(A.gamma)
    np.save(data_dir + '/evals_gamma.npy', gamma)
    
    ############################################################
    # importance sampling, needs instanton to be computed first
    
    phi = np.load(data_dir + '/inst_phi.npy')
    theta = np.load(data_dir + '/inst_theta.npy')
    lbda = np.load(data_dir + '/inst_lbda.npy')
    meanPath, samplesTimes, times, reweight, fullHist, xbins, ybins = getTransitionPathStatisticsImportanceSampling(phi, theta, lbda)
    np.save(data_dir + '/ibis_mean_eps_{}.npy'.format(eps), meanPath)
    np.save(data_dir + '/ibis_samples_eps_{}.npy'.format(eps), samplesTimes)
    np.save(data_dir + '/ibis_times_eps_{}.npy'.format(eps), times)
    np.save(data_dir + '/ibis_reweight_eps_{}.npy'.format(eps), reweight)
    np.save(data_dir + '/ibis_hist_eps_{}.npy'.format(eps), fullHist)
    np.save(data_dir + '/ibis_xbins_eps_{}.npy'.format(eps), xbins)
    np.save(data_dir + '/ibis_ybins_eps_{}.npy'.format(eps), ybins)


