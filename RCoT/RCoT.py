# TODO: which cholesky works better?
# from scipy.linalg.basic import solve_triangular
# from scipy.linalg.decomp_cholesky import cholesky
import time
from RCoT.lpb4 import lpb4
import numpy as np
import numpy.random as npr
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial import distance_matrix

# ' RCoT - tests whether x and y are conditionally independent given z. Calls RIT if z is empty.
# ' @param x Random variable x.
# ' @param y Random variable y.
# ' @param z Random variable z.
# ' @param approx Method for approximating the null distribution. Options include:
# ' "lpd4," the Lindsay-Pilla-Basak method (default),
# ' "hbe" for the Hall-Buckley-Eagleson method,
# ' @param num_f Number of features for conditioning set. Default is 25.
# ' @param num_f2 Number of features for non-conditioning sets. Default is 5.
# ' @param seed The seed for controlling random number generation. Use if you want to replicate results exactly. Default is NULL.
# ' @return A list containing the p-value \code{p} and statistic \code{Sta}
# ' @export
# ' @examples
# ' RCIT(rnorm(1000),rnorm(1000),rnorm(1000));
# '
# ' x=rnorm(10000);
# ' y=(x+rnorm(10000))^2;
# ' z=rnorm(10000);
# ' RCIT(x,y,z,seed=2);


class RCOT:
    def __init__(self, x=None, y=None, z=None, num_f=25, num_f2=5):
        self.x = x
        self.y = y
        self.z = z
        self.approx = "lpd4"
        self.num_f = num_f
        self.num_f2 = num_f2
        self.alpha = 0.05

    def matrix2(self, mat):
        if(mat.shape[0] == 1):
            return mat.T
        return mat

    def dist(self, mat):
        diststart= time.time()
        dist = distance_matrix(mat,mat)
        dist = dist[np.tril_indices(dist .shape[0],-1)]
        return np.array(dist)

    def expandgrid(self, *itrs):
        from itertools import product
        product = list(product(*itrs))
        return {'Var{}'.format(i+1): [x[i] for x in product] for i in range(len(itrs))}

    def normalize(self, x):
        if(x.std(ddof=1) > 0):
            return ((x-x.mean())/x.std(ddof=1))
        else:
            return (x-x.mean())

    def normalizeMat(self, mat):
        # if the number of rows is zero
        if(mat.shape[0] == 0):
            mat = mat.T
        mat = np.apply_along_axis(self.normalize, 0, mat)
        return mat

    def random_fourier_features(self, x, w=None, b=None, num_f=None, sigma=None, seed=None):
        featstart = time.time()
        if (num_f is None):
            num_f = 25

        # if x is a vector make it a (n,1)
        x = self.matrix2(x)
        r = x.shape[0]  # n => datapoints
        try:
            c = x.shape[1]    # D => dimension of variable
        except:
            c = 1
        if((sigma is None) or (sigma == 0)):
            sigma = 1

        if(w is None):
            # set the seed to seed
            if(seed is not None):
                npr.seed(seed)

            # Generate normal(0,1) with (num_f*c) values
            # Shape w: (num_f, c)
            # mat1 = np.array([0.407143375,-0.082946433,-0.531864996,0.375997530,0.659402372,2.098584238,0.413929300,-0.345054296,-0.809902206,-0.172774349,0.901494800,1.812771780,0.263626838,-0.769907790,-0.423588000,0.176713358,0.527971241,-0.732369358,0.919292755,-0.305950106,-2.522914162,1.244910813,0.187508545,0.368896830,0.397140381,0.249515699,-0.370206911,0.333101938,-0.872693782,-1.663996680,2.671323186,1.327697734,1.144271466,-1.692336940,-0.662809269,0.639971070,-1.345148178,-0.204515901,-1.047413260,0.862253046,-0.968340676,-0.157516137,0.202941717,0.909875126,0.058747322,-0.194787798,-1.344127808,-0.288276512,-0.732177922,0.418895242,-1.589930857,-0.497428579,1.514089693,-0.428010015,0.268634489,1.447049586,-0.649677728,1.137384395,0.321909286,-2.464546819,1.349033667,-0.873233419,-1.883680670,1.671276385,0.155927838,1.133507316,0.476693365,-0.074524124,0.917279582,0.477007669,0.026339152,1.653451141,-0.682363769,0.627757840,1.197442049,0.006848126,-0.630634922,-0.911413939,0.416048339,0.500866745,-0.854716354,-0.076653883,-0.742578015,0.360453981,0.163653745,-2.639027148,-0.182713318,-0.462752470,-0.001230553,2.975740818,-0.371817982,-0.594099092,0.085438228,0.324588935,1.229146671,-0.032959963,0.669224625,-0.455084088,1.053485663, 0.147200136])[:num_f*c]
            # mat1 = mat1.reshape(num_f,c,order='F')
            # w = (1/sigma)*mat1
            w = (1/sigma)*norm.rvs(size=(num_f*c))
            w = w.reshape(num_f, c, order='F')

            # set the seed to seed
            if(seed is not None):
                npr.seed(seed)

            # Create a row vector b of (1,r) with each value is in the range of 0 to 2*pi
            # Shape of b = (num_f,n)
            # mat2 = np.array([0.24943679,0.40272681,0.41300218,0.82630648,0.36161763,0.38695633,0.85242333,0.36338912,0.23988081,0.80964438,0.81575349,0.72413504,0.32132191,0.26886340,0.19413426,0.53087783,0.36558392,0.72362623,0.70754074,0.45836723,0.77368370,0.21385798,0.10236606,0.74187866,0.36907791,0.24600552,0.54610607,0.10645538,0.41616002,0.97929398,0.61088091,0.95699205,0.96829693,0.22630670,0.36389070,0.64923356,0.11070161,0.04423031,0.55400297,0.07435684,0.25138274,0.23333400,0.26340102,0.36885784,0.72959285,0.05182755,0.59379937,0.42187447,0.17050232,0.32351448,0.20207343,0.07066087,0.68936197,0.19694862,0.16783700,0.91276181,0.04259594,0.02395041,0.99855848,0.46649448,0.43413134,0.81925986,0.79406436,0.28507148,0.50012750,0.88732619,0.03500649,0.72745815,0.20996937,0.36801534,0.90235485,0.48449460,0.80850847,0.14072908,0.03380254,0.69850162,0.30375908,0.55768037,0.20166871,0.75252968,0.86096782,0.46792307,0.11829211,0.20294106,0.07646112,0.87190160,0.48387409,0.66052781,0.69248624,0.84302903,0.70422853,0.69536521,0.42998258,0.41419061,0.79888090,0.86357834,0.80078755,0.96816220,0.73031027,0.61615044])[:num_f]
            # b = np.repeat(2*np.pi*mat2[:,np.newaxis],r,axis=1)
            # b = repmat(2*np.pi*mat2, 1, r)
            b = npr.uniform(size=num_f)
            b = np.repeat(2*np.pi*b[:, np.newaxis], r, axis=1)
        
        feat = np.sqrt(2)*((np.cos(w[:num_f, :c] @ x.T + b[:num_f, :])).T)
        return (feat, w, b)

    def colMeans(self, vec):
        vec = np.array(vec)
        return np.mean(vec, axis=0)

    def RIT(self, x, y, num_f2=5, seed=None,r=500):
        x = np.matrix(x).T
        y = np.matrix(y).T

        if(np.std(x) == 0 or np.std(y) == 0):
            return 1   # this is P value

        x = self.matrix2(x)
        y = self.matrix2(y)

        r = x.shape[0]
        if(r > 500):
            r1 = 500
        else:
            r1 = r

        x = self.normalizeMat(x).T
        y = self.normalizeMat(y).T
        (four_x, w, b) = self.random_fourier_features(
            x, num_f=num_f2, sigma=np.median(self.dist(x[:r1, ])), seed=seed)
        (four_y, w, b) = self.random_fourier_features(
            y, num_f=num_f2, sigma=np.median(self.dist(y[:r1, ])), seed=seed)
        f_x = self.normalizeMat(four_x)
        f_y = self.normalizeMat(four_y)


        Cxy = np.cov(f_x, f_y, rowvar=False)
        Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2
        Cxy = np.round(Cxy, decimals=7)
        Sta = r*np.sum(Cxy**2)

        res_x = f_x - np.repeat(np.matrix(self.colMeans(f_x))
                                [:, np.newaxis], r, axis=1)
        res_y = f_y - np.repeat(np.matrix(self.colMeans(f_x))
                                [:, np.newaxis], r, axis=1)

        start1 = time.time()
        d = self.expandgrid(
            np.arange(0, f_x.shape[1]), np.arange(0, f_y.shape[1]))
        res = np.array(res_x[:, np.array(d['Var1'])]) * \
            np.array(res_y[:, np.array(d['Var2'])])
        res = np.matrix(res)
        Cov = 1/r * ((res.T) @ res)
        w, v = np.linalg.eig(Cov)
        w = [i for i in w if i > 0]
        if(self.approx == "lpd4"):

            w1 = w
            p = 1 - lpb4(np.array(w1), Sta)
            if(p == None or np.isnan(p)):
                from RCoT.hbe import hbe
                p = 1 - hbe(w1, Sta)

        return (p, Sta)

    def rcot(self, x, y, z=None, num_f=25, num_f2=5, seed=None,r=500):
        start = time.time()
        x = np.matrix(x).T
        y = np.matrix(y).T
        # Unconditional Testing
        if(len(z) == 0 or z == None):
            (p, Sta) = self.RIT(x, y, num_f2, seed,r)
            return (None, Sta, p)
        
        z = np.matrix(z).T
        x = self.matrix2(x)
        y = self.matrix2(y)
        z = self.matrix2(z)

        # Convert later to lamnda function
        z1 = []
        try:
            c = z.shape[1]
        except:
            c = 1

        for i in range(c):
            if(z[:, i].std() > 0):
                z1.append(z[:, i])

        #z = z1[0]
        z = self.matrix2(z)
        try:
            d = z.shape[1]    # D => dimension of variable
        except:
            d = 1
        # Unconditional Testing
        if(len(z) == 0 or z.any() == None):
            (p, Sta) = self.RIT(x, y, num_f2, seed,r)
            return (None, Sta, p)

        # Sta - test statistic -> s
        # if sd of x or sd of y == 0 then x and y are independent
        if (x.std() == 0 or y.std() == 0):
            # p=1 and Sta=0
            out = (1, 0)
            return(out)

        # make it explicit as maxData
        if (r >  x.shape[0]):
            r1 =  x.shape[0]
        else:
            r1 = r
        # Normalize = making it as mean =0 and std= 1
        x = self.normalizeMat(x).T
        y = self.normalizeMat(y).T
        if(d == 1):
            z = self.normalizeMat(z).T
        else:
            z = self.normalizeMat(z)
        (four_z, w, b) = self.random_fourier_features(
            z[:, :d], num_f=num_f, sigma=np.median(self.dist(z[:r1, ])), seed=seed)

        (four_x, w, b) = self.random_fourier_features(
            x, num_f=num_f2, sigma=np.median(self.dist(x[:r1, ])), seed=seed)

        (four_y, w, b) = self.random_fourier_features(
            y, num_f=num_f2, sigma=np.median(self.dist(y[:r1, ])), seed=seed)
        f_x = self.normalizeMat(four_x)
        f_y = self.normalizeMat(four_y)  # n,numf2
        f_z = self.normalizeMat(four_z)  # n,numf

        # Next few lines will be Equation2 from RCoT paper
        Cxy = np.cov(f_x, f_y, rowvar=False)  # 2*numf2,2*numf2

        Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2

        Cxy = np.round(Cxy, decimals=7)

        Czz = np.cov(f_z, rowvar=False)  # numf,numf

        # Czz = np.round(Czz, decimals=7)

        I = np.eye(num_f)
        L = cholesky((Czz + (np.eye(num_f) * 1e-10)), lower=True)
        L_inv = solve_triangular(L, I, lower=True)
        i_Czz = L_inv.T.dot(L_inv)  # numf,numf

        Cxz = np.cov(f_x, f_z, rowvar=False)[:num_f2, num_f2:]  # numf2,numf

        Czy = np.cov(f_z, f_y, rowvar=False)[:num_f, num_f:]  # numf,numf2

        z_i_Czz = f_z @ i_Czz  # (n,numf) * (numf,numf)
        e_x_z = z_i_Czz @ Cxz.T  # n,numf
        e_y_z = z_i_Czz @ Czy

        # approximate null distributions

        # residual of fourier after it removes the effect of ??
        res_x = f_x-e_x_z
        res_y = f_y-e_y_z

        # if (num_f2 == 1):
        #     approx = "hbe"

        matmul = (Cxz @ (i_Czz @ Czy))
        matmul = np.round(matmul, decimals=7)
        Cxy_z = Cxy-matmul  # less accurate for permutation testing

        Sta = r * np.sum(Cxy_z**2)

        d = self.expandgrid(
            np.arange(0, f_x.shape[1]), np.arange(0, f_y.shape[1]))
        res = np.array(res_x[:, np.array(d['Var1'])]) * \
            np.array(res_y[:, np.array(d['Var2'])])
        res = np.matrix(res)
        Cov = 1/r * ((res.T) @ res)

        w, v = np.linalg.eigh(Cov)
        w = [i for i in w if i > 0]

        if(self.approx == "lpd4"):
            # from lpb4 import lpb4
            w1 = w
            p = 1 - lpb4(np.array(w1), Sta)
            if(p == None or np.isnan(p)):
                from hbe import hbe
                p = 1 - hbe(w1, Sta)
        return (Cxy_z, Sta, p)
    
    def independence(self, x, y, z=None, num_f=25, num_f2=5, seed=None,r=500):
        (Cxy,Sta,p) = self.rcot(x, y, z, num_f, num_f2, seed,r)
        dependence =  max(0, (.5 + (self.alpha-p)/(self.alpha*2)), (.5 - (p-self.alpha)/(2*(1-self.alpha))))
        return (1-dependence)

    def dependence(self, x, y, z=None, num_f=25, num_f2=5, seed=None,r=500):
        independence = self.independence(x, y, z, num_f, num_f2, seed,r)
        return 1-independence

# main = time.time()
# from pandas import read_csv
# data = read_csv("./indCalibrationDat.csv")
# x = list(data['A'])
# z = list(data['B'])
# y = list(data['C'])
# rs = RCOT(x, y, z)
# (Cxy_z, Sta, p) = rs.independence(x, y, z)
# print("Time for whole:", time.time()-main)
