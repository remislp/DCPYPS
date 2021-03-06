import math
from scipy import stats
import numpy as np

class Equation(object):
    def __init__(self):
        """        """
        self.eqname = None
        self.ncomp = 1
        self.pars = None
        self.fixed = []
        self.names = []
        self.data = None
        self.guess = None
        self._theta = None
        self.normalised = False
        
    def equation(self, x, coeff):
        ''' '''
        pass
    
    def calculate_random(self, x, sd):
        """ """
        if isinstance(x, float):
            return np.random.normal(self.equation(x, self.pars), sd, 1)[0]
        elif isinstance(x, list) or isinstance(x, np.ndarray):
            resp = []
            for each in x:
                resp.append(np.random.normal(self.equation(each, self.pars), sd, 1)[0])
            return np.array(resp)
        
    def to_fit(self, x, theta):
        self._set_theta(theta)
        return self.equation(x, self.pars)
    
    def normalise(self, data):
        pass
    
    def _set_theta(self, theta):
        for each in np.nonzero(self.fixed)[0]:   
            theta = np.insert(theta, each, self.pars[each])
        self.pars = theta
    def _get_theta(self):
        theta = self.pars[np.nonzero(np.invert(self.fixed))[0]]
        if isinstance(theta, float):
            theta = np.array([theta])
        return theta
    theta = property(_get_theta, _set_theta)
    
    def __repr__(self):
        txt = "equation " + self.eqname + "\n"
        for name, par in zip(self.names, self.pars):
            txt += "{} = {}\n".format(name, par)
        return txt
    

class GHK(Equation):
    """Goldman-Hodgkin-Katz equation for bi-ionic condition"""
    def __init__(self, eqname, pars=None):
        """
        pars = [r]
        """
        self.eqname = eqname
        self.pars = pars
        self.fixed = [False, True, True, True]
        self.names = ['r', 'totOut', 'In1', 'In2']
        
    def equation(self, x, pars):
        '''
        The GHK equation.
        '''
        return 25 * np.log( (x + pars[0] * (pars[1] - x)) / (pars[2] + pars[0] * pars[3]) )
   
    def propose_guesses(self, data=None):
        '''
        '''
        self.guess = self.pars.copy()
        
    def calculate_plot(self, X, coeff):
        plX = np.linspace(np.floor(np.amin(X)), np.ceil(np.amax(X)), 100)
        plY = self.equation(plX, coeff)
        return plX, plY

    
class Linear(Equation):
    def __init__(self, eqname, pars=None):
        """
        pars = [a, b]
        """
        self.eqname = eqname
        self.ncomp = 1
        self.pars = pars
        self.fixed = [False, False]
        self.names = ['a', 'b']
        
    def equation(self, x, coeff):
        '''
        The linear equation.
        '''
        return coeff[0] + coeff[1] * x 
   
    def propose_guesses(self, data):
        '''
        Calculate the initial guesses for fitting with Linear equation.
        '''
        #if self.Component == 1:
        slope, intercept, r, p, stderr = stats.linregress(data.X, data.Y)
        self.guess = np.array([intercept, slope])
        self.pars = self.guess.copy()
        
    def calculate_plot(self, X, coeff):
        plotX = np.linspace(np.floor(np.amin(X) - 1),
            np.ceil(np.amax(X) + 1), 100)
        plotY = self.equation(plotX, coeff)
        return plotX, plotY

    
class ExponentialPDF(Equation):
    def __init__(self, eqname='ExpPDF', pars=None, ncomp=1):
        self.eqname = eqname
        self.ncomp = ncomp
        self.pars = pars
        self.fixed = [False, False] * ncomp
        self.fixed[-1] = True
        self.names = ['tau', 'area']
    
    def equation(self, theta, X):
        tau, area = self.theta_unsqueeze(theta)
        X = np.asarray(X)
        y = np.array([])
        for t in np.nditer(X):
            y = np.append(y, np.sum((area / tau) * np.exp(-t / tau)))
        return y

    def theta_unsqueeze(self, theta):
        theta = np.asarray(theta)
        tau, area = np.split(theta, [int(math.ceil(len(theta) / 2))])
        area = np.append(area, 1 - np.sum(area))
        return tau, area

class Exponential(Equation):
    def __init__(self, eqname, pars=None):
        """
        pars = [a, b]
        """
        self.eqname = eqname
        self.ncomp = 1
        self.pars = pars
        self.fixed = [False, False]
        self.names = ['area', 'tau']
        
    def equation(self, x, coeff):
        '''
        The exponential equation.
        '''
        return coeff[0] / coeff[1] + math.exp(-x /coeff[1]) 

    def calculate_plot(self, X, coeff):
        plotX = np.linspace(np.floor(np.amin(X) - 1),
            np.ceil(np.amax(X) + 1), 100)
        plotY = self.equation(plotX, coeff)
        return plotX, plotY


class Hill(Equation):
    def __init__(self, eqname, pars=None):
        """
        pars = [Ymin, Ymax, EC50, nH]
        """
        self.eqname = eqname
        self.ncomp = 1
        if pars:
            self.pars = pars
        else:
            self.pars = [0.0, 100.0, 10.0, 1.0]
        if eqname == 'Hill':
            self.fixed = [True, False, False, False]
            self.names = ['Ymin', 'Ymax', 'EC50', 'nH  ']
        if eqname == 'Langmuir':
            self.fixed = [True, False, False, True]
            self.names = ['Ymin', 'Ymax', 'EC50']
        self.normpars = None
        self.normalised = False
        
    def equation(self, conc, coeff):
        '''
        The hill equation.
        '''
        return (coeff[0] + ((coeff[1] - coeff[0]) * (conc / coeff[2]) ** coeff[3]) / 
            (1 + (conc / coeff[2]) ** coeff[3]))
            
    def normalise(self, data):
        '''
        Nomalise Y to the fitted maximum.
        '''
        # Nomalise the coefficients by fixing the Y(0) and Ymax
        self.normpars = self.pars.copy()
        self.normpars[0], self.normpars[1] = 0, 1
        #data.normY = (data.Y - self.pars[0]) / self.pars[1]
        data.normY = data.Y / self.pars[1]
        data.normS = data.S / self.pars[1]
        self.normalised = True
    
    def propose_guesses(self, data):
        '''
        Calculate the initial guesses for fitting with Hill equation.
        '''
        #if self.Component == 1:
        self.guess = np.empty(4)
        if data.increase: # Response increases with concentration
            # Determine Y(0)
            if self.fixed[0]:
                self.guess[0] = 0
            else:
                self.guess[0] = np.mean(data.Y[data.X == data.X[0]])
            if self.fixed[1]:
                self.guess[1] = 1
            else:
                # Determine Ymax
                self.guess[1] = np.mean(data.Y[data.X == data.X[-1]])# - self.guess[0]
        else: # Response decreases with concentration
            # Determine Y(0)
            if self.fixed[0]:
                self.guess[0] = 0
            else:
                self.guess[0] = np.mean(data.Y[data.X == data.X[-1]])
            # Determine Ymin
            self.guess[1] = np.mean(data.Y[data.X == data.X[0]])# - self.guess[0]
        # Determine Kr
        self.guess[2] = 10 ** ((np.log10(data.X[0]) + np.log10(data.X[-1])) / 2)
        # Determine nH  
        LinRegressX = np.log10(data.X[data.Y < np.amax(data.Y)]) - np.log10(self.guess[2])
        ratio = data.Y[data.Y < np.amax(data.Y)] / np.amax(data.Y)
        LinRegressY = np.log10(ratio / (1 - ratio))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            LinRegressX, LinRegressY)
        if math.isnan(slope):
            self.guess[3] = 1.0 if data.increase else -1.0
        else:
            self.guess[3] = slope
        if self.eqname == 'Langmuir':
            self.guess[3] = slope / math.fabs(slope)
#        elif self.Component == 2:
#            print 'Two Components fitting is not completed.'
#            sys.exit(0)
        self.pars = self.guess.copy()
        
    def calculate_plot(self, X, coeff):
        plotX = 10 ** np.linspace(np.floor(np.amin(np.log10(X)) - 1),
            np.ceil(np.amax(np.log10(X)) + 1), 100)
        plotY = self.equation(plotX, coeff)
        return plotX, plotY

