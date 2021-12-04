#@author:  Namita Kharat,Manish Bhattarai
from scipy.stats import wilcoxon
from . import config
from .dist_clustering import *
from .pyDRESCAL import *
from .plot_results import *

class sample():
    """
    Generates perturbed version of data based on sampling distribution.

    Args:
        data (ndarray, sparse matrix): Array of which to find a perturbation.
        noise_var (float): The perturbation amount.
        method (str) : Method for sampling (uniform/poisson)
        seed (float),optional : Set seed for random data generation
    """


    @comm_timing()
    def __init__(self, data, noise_var, method, params,seed=None):
        self.np = params.np
        self.X = data
        self.noise_var = noise_var
        self.seed = seed
        if self.seed != None:
            self.np.random.seed(self.seed)
        self.method = method
        self.X_per = 0

    @comm_timing()
    def randM(self):
        """
        Multiplies each element of X by a uniform random number in (1-epsilon, 1+epsilon).
        """

        M = 2 * self.noise_var * self.np.random.random_sample(self.X.shape).astype(self.X.dtype) + self.noise_var
        M = M + 1
        self.X_per = self.np.multiply(self.X, M)

    @comm_timing()
    def poisson(self):
        """Resamples each element of a matrix from a Poisson distribution with the mean set by that element. Y_{i,j} = Poisson(X_{i,j}"""

        self.X_per = self.np.random.poisson(self.X).astype(self.X.dtype)

    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform resampling on data

        Returns
        -------
        X_per : ndarry
           Perturbed version of data
        """

        if self.method == 'uniform':
            self.randM()
        elif self.method == 'poisson':
            self.poisson()
        return self.X_per


class pyDRESCALk():
    r"""
    Performs the distributed RESCAL decomposition with custom clustering for estimating hidden factors k

    Parameters:
        A_ij (ndarray) : Distributed Data
        factors (tuple), optional : Distributed factors W and H
        params (class): Class which comprises following attributes
        params.init (str) : RESCAL initialization(rand/nnsvd)
        params.comm1 (object): Global Communicator
        params.comm (object): Modified communicator object
        params.k (int) : Rank for decomposition
        params.m (int) : Global dimensions m
        params.n (int) : Global dimensions n
        params.p_r  (int): Cartesian grid row count
        params.p_c  (int): Cartesian grid column count
        params.row_comm (object) : Sub communicator along row
        params.col_comm (object) : Sub communicator along columns
        params.A_update (bool) : flag to set W update True/False
        params.norm (str): RESCAL norm to be minimized
        params.method(str): RESCAL optimization method
        params.eps (float) : Epsilon value
        params.verbose (bool) : Flag to enable/disable display results
        params.save_factors (bool) : Flag to enable/disable saving computed factors
        params.perturbations (int) : Number of Perturbations for clustering
        params.noise_var (float) : Set noise variance for perturbing the data
        params.sill_thr (float) : Set the sillhouette threshold for estimating K with p-test
        params.start_k (int) : Starting range for Feature search K
        params.end_k (int) : Ending range for Feature search K"""

    @comm_timing()
    def __init__(self, X_ijk, factors=None, params=None):
        self.X_ijk = X_ijk
        self.local_m, self.local_n, self.local_n = len(self.X_ijk),self.X_ijk[0].shape[0],self.X_ijk[0].shape[1]
        self.params = params
        self.np = self.params.np
        self.comm1 = self.params.comm1
        self.rank = self.comm1.rank
        self.p_r, self.p_c = self.params.p_r, self.params.p_c
        self.fpath = self.params.fpath
        self.fname = self.params.fname
        #self.fname = "Testrescalk"
        self.p = self.p_r * self.p_c
        if self.p_r != 1 and self.p_c != 1:
            self.topo = '2d'
        else:
            self.topo = '1d'
        self.sampling = var_init(self.params,'sampling',default='uniform')
        self.perturbations = var_init(self.params,'perturbations',default=10)
        self.noise_var = var_init(self.params,'noise_var',default=.03)
        self.Rall = 0
        self.Aall = 0
        self.recon_err = 0
        self.AvgR = 0
        self.AvgG = 0
        self.col_err = 0
        self.clusterSilhouetteCoefficients, self.avgSilhouetteCoefficients = 0, 0
        self.L_errDist = 0
        self.avgErr = 0
        self.start_k = self.params.start_k  # ['start_k']
        self.end_k = self.params.end_k  # ['end_k']
        self.step_k = var_init(self.params,'step_k',default=1)
        self.verbose = var_init(params,'verbose',default=True)


    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform distributed RESCAL decomposition and then custom clustering to estimate k

        Returns
        -------
        nopt : int
           Estimated value of latent features
        """
        SILL_MIN = []
        SILL_AVG = []
        errRegres = []
        errRegresTol = []
        RECON = []
        RECON1 = []
        self.params.results_paths = self.params.results_path +self.params.fname + '/'
        if self.rank == 0:
            try: os.makedirs(self.params.results_paths)
            except: pass
        for self.k in range(self.start_k, self.end_k + 1,self.step_k):
            self.params.k = self.k
            self.pyrescalk_per_k()
            SILL_MIN.append(self.np.around(self.np.min(self.clusterSilhouetteCoefficients), 2))
            SILL_AVG.append(self.np.around(self.np.mean(self.clusterSilhouetteCoefficients), 2))
            errRegres.append([self.col_err])
            errRegresTol.append([self.recon_err])
            RECON.append(self.L_errDist)
            RECON1.append(self.avgErr)
        if self.rank==0:
            plot_results_paper(self.start_k, self.end_k,self.step_k, RECON, SILL_AVG, SILL_MIN, self.params.results_path, self.fname)


    @comm_timing()
    def pyrescalk_per_k(self):
        """Performs RESCAL decomposition and clustering for each k to estimate silhouette statistics"""
        self.params.results_paths = self.params.results_path+ str(self.k) + '/'
        if self.rank == 0:
            try: os.makedirs(self.params.results_paths)
            except: pass
        results = []
        if self.rank == 0: print('*************Computing for k=', self.k, '************')
        for i in range(self.perturbations):
            if self.rank == 0: print('Current perturbation =', i)
            self.params.perturbation = i
            data = sample(data=self.X_ijk, noise_var=self.noise_var, method=self.sampling,params=self.params, seed=self.rank*1000+i*100).fit()
            self.params.A_update = True
            results.append(pyDRESCAL(data, factors=None, params=self.params).fit())
        self.Aall = self.np.stack([results[i][0] for i in range(self.perturbations)],axis=-1)
        #self.Aall = self.Aall.reshape(self.Aall.shape[0], self.k, self.perturbations, order='F')      #n x k x perturbations
        self.Rall = self.np.stack([results[i][2] for i in range(self.perturbations)],axis=-1)
        #self.Rall = self.Rall.reshape(results[0][2].shape[0], self.k, self.Rall.shape[1], self.perturbations)    #m x k x k x perturbations
        self.recon_err = [results[i][3] for i in range(self.perturbations)]
        [processAvg, processSTD, self.Rall, self.clusterSilhouetteCoefficients, self.avgSilhouetteCoefficients,
                idx] = custom_clustering(self.Aall, self.Rall, self.params).fit()
        self.AvgR = self.np.median(self.Rall, axis=-1)
        self.AvgA = processAvg
        self.params.A_update = False
        regressH = pyDRESCAL(self.X_ijk, factors=[self.AvgA, self.AvgR], params=self.params)
        self.AvgA, self.AvgA_j, self.AvgR, self.L_errDist = regressH.fit()
        self.avgErr = np.mean(self.recon_err)
        cluster_stats = {'clusterSilhouetteCoefficients': self.clusterSilhouetteCoefficients,
                         'avgSilhouetteCoefficients': self.avgSilhouetteCoefficients, \
                          'avgErr': self.avgErr, 'recon_err': self.recon_err,'L_errDist':self.L_errDist}
        data_writer = data_write(self.params)
        data_writer.save_factors([self.AvgA, self.AvgR], reg=True)
        data_writer.save_cluster_results(cluster_stats)
