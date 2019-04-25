"""
Collection of functions useful for plotting in DCPYPS.
"""
__author__ = "Remis"
__date__ = "$22-Feb-2016 09:13:22$"

import math
import numpy as np
from numpy import linalg as nplin
import matplotlib.pyplot as plt

from HJCFIT.likelihood import QMatrix
from HJCFIT.likelihood import missed_events_pdf, ideal_pdf, IdealG, eig

import dcpyps.dataset
from scalcs import scalcslib as scl
from scalcs import qmatlib as qml
from scalcs import pdfs
from scalcs import scplotlib as scpl


def xlog_hist_data(ax, X, tres, shut=True, unit='s'):
    """
    Plot dwell time histogram in log x and square root y.
    """
    
    xout, yout, dx = prepare_xlog_hist(X, tres)
    ax.semilogx(xout, np.sqrt(yout))
    ax.set_xlabel('Apparent {0} periods ({1})'.
        format('shut' if shut else 'open', unit))
    ax.set_ylabel('Square root of frequency density')

def xlog_hist_data_multi(ax, X, tres, names, shut=True, unit='s', density=False):
    """
    Plot multiple dwell time histograms in log x and square root y.
    """
    for x, tr, name in zip(X, tres, names):
        dfactor = 1.0
        #if density: dfactor = 1.0 / len(x)
        xout, yout, dx = prepare_xlog_hist(x, tr)
        ax.semilogx(xout, np.sqrt(dfactor * yout), label=name)
        ax.set_xlabel('Apparent {0} periods ({1})'.
            format('shut' if shut else 'open', unit))
        ax.set_ylabel('Square root of frequency density')
    ax.legend(loc=1, borderaxespad=0.)


    
def xlog_hist_HJC_fit(ax, tres, X=None, pdf=None, ipdf=None, iscale=None, 
                       shut=True, tcrit=None, legend=True, unit='s'):
    """
    Plot dwell time histogram and predicted pdfs (ideal and corrected for the 
    missed events) in log x and square root y.
    """

    scale = 1.0
    if X is not None:
        tout, yout, dt = prepare_xlog_hist(X, tres)
        ax.semilogx(tout, np.sqrt(yout))
        scale = len(X) * math.log10(dt) * math.log(10)
        
    if pdf and ipdf:
        if shut:
            t = np.logspace(math.log10(tres), math.log10(tcrit), 512)
            ax.semilogx(t, np.sqrt(t * ipdf(t) * scale * iscale), '--r', 
                label='Ideal distribution')
            ax.semilogx(t, np.sqrt(t * pdf(t) * scale), '-b', 
                label='Corrected distribution')
            t = np.logspace(math.log10(tcrit), math.log10(max(X) *2), 512)
            ax.semilogx(t, np.sqrt(t * pdf(t) * scale), '--b')
            ax.semilogx(t, np.sqrt(t * ipdf(t) * scale * iscale), '--r')
            ax.axvline(x=tcrit, color='g')
        else:
            t = np.logspace(math.log10(tres), math.log10(2 * max(X)), 512)
            ax.semilogx(t, np.sqrt(t * ipdf(t) * scale * iscale), '--r',
                label='Ideal distribution')
            ax.semilogx(t, np.sqrt(t * pdf(t) * scale), '-b', 
                label='Corrected distribution')
                
    ax.set_xlabel('Apparent {0} periods ({1})'.
        format('shut' if shut else 'open', unit))
    ax.set_ylabel('Square root of frequency density')
    if legend: ax.legend(loc=(1 if shut else 3))
    
def xlog_hist_EXP_fit(ax, tres, X=None, pdf=None, pars=None, shut=True, 
                      tcrit=None, unit='s'):
    """
    Plot dwell time histogram and multi-exponential pdf with single 
    components in log x and square root y.
    """
    
    theta = np.asarray(pars)
    tau, area = np.split(theta, [int(math.ceil(len(theta) / 2))])
    area = np.append(area, 1 - np.sum(area))

    scale = 1.0
    if X is not None:
        tout, yout, dt = prepare_xlog_hist(X, tres)
        ax.semilogx(tout, np.sqrt(yout))
        scale = (len(X) * math.log10(dt) * math.log(10) *
            (1 / np.sum(area * np.exp(-tres / tau))))
        
    t = np.logspace(math.log10(tres), math.log10(2 * max(X)), 512)
    ax.plot(t, np.sqrt(scale * t * pdf(t)), '-b')
    for ta, ar in zip(tau, area):
        ax.plot(t, np.sqrt(scale * t * (ar / ta) * np.exp(-t / ta)), '--b')
        
    if tcrit is not None:
        tcrit = np.asarray(tcrit)
        for tc in tcrit:
            ax.axvline(x=tc, color='g')
        
    ax.set_xlabel('Apparent {0} periods ({1})'.
        format('shut' if shut else 'open', unit))
    ax.set_ylabel('Square root of frequency density')
    
def prepare_xlog_hist(X, tres):
    """
    Prepare data points for x-log histogram to plot in matplotlib
    
    eg. 
    xout, yout, dx = dcplots.prepare_xlog_hist(intervals, resolution)
    plt.plot(xout, yout)
    plt.xscale('log')
    
    Parameters
    ----------
    X :  1-D array or sequence of scalar
    tres : float
        Temporal resolution, shortest resolvable time interval. It is
        histogram's starting point.

    Returns
    -------
    xout, yout :  list of scalar
        x and y values to plot histogram.
    dx : float
        Histogram bin width.

    """

    # Defines bin width and number of bins.
    # Number of bins/decade
    n = len(X)
    if (n <= 300): nbdec = 5
    if (n > 300) and (n <= 1000): nbdec = 8
    if (n > 1000) and (n <= 3000): nbdec = 10
    if (n > 3000): nbdec = 12
    dx = math.exp(math.log(10.0) / float(nbdec))
    xstart = tres    # histogramm starts at
    xmax = max(X)
    # round up maximum value, so get Xmax for distribution
    xend = math.exp(math.ceil(math.log(xmax)))
    nbin = int(math.log(xend / xstart) / math.log(dx))
    
#    xaxis = np.arange(xstart, xend, dx)

    # Make bins.
    xaxis = np.zeros(nbin+1)
    xaxis[0] = xstart
    # For log scale.
    for i in range(1, nbin+1):
        xaxis[i] = xstart * (dx**i)

    # Sorts data into bins.
    freq = np.zeros(nbin)
    for i in range(n):
        for j in range(nbin):
            if X[i] >= xaxis[j] and X[i] < xaxis[j+1]:
                freq[j] = freq[j] + 1

    xout = np.zeros((nbin + 1) * 2)
    yout = np.zeros((nbin + 1) * 2)

    xout[0] = xaxis[0]
    yout[0] = 0
    for i in range(0, nbin):
        xout[2*i+1] = xaxis[i]
        xout[2*i+2] = xaxis[i+1]
        yout[2*i+1] = freq[i]
        yout[2*i+2] = freq[i]
    xout[-1] = xaxis[-1]
    yout[-1] = 0

    return xout, yout, dx


def moving_average(x, n):
    """
    Compute an n period moving average.
    """
    x = np.asarray(x)
    weights = np.ones(n)
    weights /= weights.sum()
    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def scalefac(tres, matrix, phiA):
    """ Scale factor for ideal pdf. 
    Note that to properly overlay ideal and missed-event corrected pdfs-
    ideal pdf has to be scaled (need to renormailse to 1 the area under 
    pdf from tres). """
    k = matrix.shape[0]
    if k == 1:
        return 1 # / ((1 / eigs) * np.exp(-tres * eigs))
    else:
        eigs, M = eig(-matrix)
        N = nplin.inv(M)
        A, w = np.zeros((k, k, k)), np.zeros(k)
        for i in range(k):
            A[i] = np.dot(M[:, i].reshape(k, 1), N[i].reshape(1, k))
        for i in range(k):
            w[i] = np.dot(np.dot(np.dot(phiA, A[i]), (-matrix)), np.ones((k, 1)))
        return 1 / np.sum((w / eigs) * np.exp(-tres * eigs))

def plot_mean_open_next_shut(ax, mec, rates, rec, line='b-'):
    mec.theta_unsqueeze(rates)
    mec.set_eff('c', rec.conc)
    sht, mp, mn = scpl.mean_open_next_shut(mec, rec.tres)
    ax.plot(sht, mp, 'b-')
    
def plot_hist(ax, X, tres, tcrit=np.inf):
    """
    """
    tout, yout, dt = prepare_xlog_hist(
        np.array(X)[(np.array(X)<math.fabs(tcrit))], tres)
    scale = len(X) * math.log10(dt) * math.log(10)
    ax.semilogx(tout, np.sqrt(yout))
    return scale 

def add_pdfs(ax, mec, random_rates, conc, tres, scale, tcrit=np.inf, color='b'):
    for rates in random_rates:
        mec.theta_unsqueeze(rates)
        mec.set_eff('c', conc)
        if tcrit == np.inf:
            t, ipdf, epdf = scpl.open_time_pdf(mec, tres)
        else:
            t, ipdf, epdf = scpl.shut_time_pdf(mec, tres, tmax=tcrit)
        ax.semilogx(t, np.sqrt(t * epdf * scale), color+'-')
        
def adjust_limits_titles(ax, conc, xmin, xmax, is_shut=False):
    ax.set_xlim(xmin, xmax)
#    ax.text(0.5, 0.2, 
#        'concentration = {0:g} mM'.format(round(float(conc*1000), 6)))
    ax.set_title('concentration = {0:g} mM'.format(round(float(conc*1000), 6)))
    labels = ax.get_yticks()**2
    ax.set_yticklabels([str(int(label)) for label in labels])
    if is_shut:
        ax.set_xlabel('Apparent shut period, s')
    else:
        ax.set_xlabel('Apparent open period, s')
    ax.set_ylabel('Frequency density (square root)')
    
def composite_fit_results(mec, recs,ratesM=None, time_range=None, random_rates=None):
    nrecs = len(recs)
    fig = plt.figure(figsize = (15,4*nrecs))
    for i in range(nrecs):

        # Plot apparent open period histogram
        ax1 = fig.add_subplot(nrecs, 3, i*3+1)
        
        scale = plot_hist(ax1, recs[i].opint, recs[i].tres)
        
        if random_rates is not None:
            add_pdfs(ax1, mec, random_rates, recs[i].conc, recs[i].tres, scale)
        
        add_pdfs(ax1, mec, [ratesM], recs[i].conc, recs[i].tres, scale, color='r')
        
        adjust_limits_titles(ax1, recs[i].conc, 10e-6, 10, is_shut=False)

        # Plot apparent shut period histogram
        ax2 = fig.add_subplot(nrecs, 3, i*3+2)
        scale = plot_hist(ax2, recs[i].shint, recs[i].tres, tcrit=math.fabs(recs[i].tcrit))
        if random_rates is not None:
            add_pdfs(ax2, mec, random_rates, recs[i].conc, recs[i].tres, scale, 
                     tcrit=math.fabs(recs[i].tcrit))
        add_pdfs(ax2, mec, [ratesM], recs[i].conc, recs[i].tres, scale, 
                 tcrit=math.fabs(recs[i].tcrit), color='r')
        adjust_limits_titles(ax2, recs[i].conc, 1e-5, 1, is_shut=True)

        # correlation plots
        #ax3 = fig.add_subplot(3, nrecs, 2*nrecs+i+1)
        ax3 = fig.add_subplot(nrecs, 3, i*3+3)
        mean_shut, mean_open, error = dcpyps.dataset.mean_open_shut_correlation(recs[i], time_range)
        ax3.errorbar(mean_shut, mean_open, yerr=error,  fmt='o')
        if random_rates is not None:
            for rates in random_rates:
                plot_mean_open_next_shut(ax3, mec, rates, recs[i], line='b-')
        plot_mean_open_next_shut(ax3, mec, ratesM, recs[i], line='r-')
        ax3.set_xscale('log')
        ax3.set_xlabel('Mean apparent shut time, s')
        ax3.set_ylabel('Mean adjacent open time, s')
        ax3.set_ylim(0, 10e-3)
        ax3.set_xlim(10e-6, 10)
        labels = ax3.get_xticks()
        labels = [int(label) if label >= 1 else label for label in labels]
        ax3.set_xticklabels([str(label) for label in labels])

    fig.tight_layout()
    plt.show()
    
def figure_HJCFIT(recs, mec):
    fig = plt.figure(figsize=(15,4*len(recs))) #(12,15))
    for i in range(len(recs)):
        mec.set_eff('c', recs[i].conc)
        qmatrix = QMatrix(mec.Q, mec.kA)
        idealG = IdealG(qmatrix)

        # Plot apparent open period histogram
        ax1 = fig.add_subplot(len(recs), 2, 2*i+1)

        ipdf = ideal_pdf(qmatrix, shut=False) 

        iscale = scalefac(recs[i].tres, qmatrix.aa, idealG.initial_vectors)

        epdf = missed_events_pdf(qmatrix, recs[i].tres, nmax=2, shut=False)
        xlog_hist_HJC_fit(ax1, recs[i].tres, recs[i].opint,
                                   epdf, ipdf, iscale, shut=False)
        ax1.set_title('concentration = {0:3f} mM'.format(recs[i].conc*1000))
        ax1.set_xlim(10e-6, 1)

        # Plot apparent shut period histogram
        ax2 = fig.add_subplot(len(recs), 2, 2*i+2)
        ipdf = ideal_pdf(qmatrix, shut=True)
        iscale = scalefac(recs[i].tres, qmatrix.ff, idealG.final_vectors)
        epdf = missed_events_pdf(qmatrix, recs[i].tres, nmax=2, shut=True)
        xlog_hist_HJC_fit(ax2, recs[i].tres, recs[i].shint,
                                   epdf, ipdf, iscale, tcrit=math.fabs(recs[i].tcrit))
        ax2.set_title('concentration = {0:6f} mM'.format(recs[i].conc*1000))
        ax2.set_xlim(10e-6, 1)

    fig.tight_layout()
