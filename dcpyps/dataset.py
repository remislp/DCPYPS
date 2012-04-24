#! /usr/bin/python

import math

import numpy as np
from scipy.special import erf

class TimeSeries(object):
    """
    A wrapper over a list of time intervals from idealised single channel
    record.
    """

    def __init__(self, filename, header, itint, iampl=None, iprops=None):
        
        self.filename = filename
        self.header = header
        self.itint = itint
        self.iampl = iampl
        self.iprops = iprops
        self.calfac = self.header['calfac2']

    def print_all_record(self):
        for i in range(len(self.itint)):
            print i, self.itint[i], self.iampl[i], self.iprops[i]

    def print_resolved_intervals(self):
        print('\n#########\nList of resolved intervals:\n')
        for i in range(len(self.rtint)):
            print i, self.rtint[i], self.rampl[i], self.rprops[i]
            if (self.rampl[i] == 0) and (self.rtint > (self.tcrit * 1000)):
                print ('\n')
        print('\n###################\n\n')
            
    def impose_resolution(self, tres):
        """
        Impose time resolution.
        """

        # Time intervals in SCN files are in millisec, thus, need tres in
        # millisec too.
        tres = tres * 1000

        # If last interval is shut, then set it as unusable.
        if self.iampl[-1] == 0:
            self.iprops[-1] = 8

        # Check for negative intervals and set them unusable.
        # Skip those already set unusable.
        for i in range(len(self.itint)):
            if (self.itint[i] < 0) and (self.iprops[i] != 8):
                self.iprops[i] = 8
                #print '\n interval %d set unusable.'

        # IF THE FIRST INTERVAL IS BOTH USABLE AND RESOLVABLE
        # THIS IS STARTING POINT. IF NOT LOOK FOR FIRST INTERVAL THAT
        # IS BOTH, AND IS PRECEDED BY AN RESOLVABLE INTERVAL TOO (OTHERWISE
        # ITS START WILL BE DEFINED ONLY BY THE POSITION OF THE PRECEDING
        # UNRESOLVABLE INTERVAL AND SO WILL BE UNRELIABLE)

        n = 0
        firstResolved = False
        if (self.itint[n] > tres) and (self.iprops[n] != 8):
            firstResolved = True    # first interval is usable and resolvable
            #print 'first usable interval is ', n

        while not firstResolved:
            n += 1
            if ((self.itint[n] > tres) and (self.iprops[n] != 8)): # and
                #(self.itint[n-1] > tres) and (self.iprops[n-1] != 8)):
                firstResolved = True    # first interval is usable and resolvable
                #print 'first usable interval is ', n
            #else: n += 1

        # NOW START TO LOOK FOR UNRESOLVABLE INTERVALS

        # (1) A concantenated shut period starts with a good, resolvable shutting
        #     and ends when first good resolvable opening found.
        #     Length of concat shut period=sum of all durations before the resol opening
        #     Amplitude of concat shut period=0
        # (2) A concantenated open period starts with a good, resolvable opening
        #     and ends when first good resolvable interval is found that
        #     has a different amplitude (either shut, or open but diff amplitude).
        #     Length of concat open period=sum of all concatenated durations
        #     Amplitude of concat open period weighted mean amp of all concat intervals
        # Simpler to have separate code for shut groups and for open groups. If first
        # interval of group is shut then set shutint=true.

        # First interval of any concat group must be good and resolvable so
        # insert warning to check this

        # First interval in each concatenated group must be resolvable, but may
        # be bad (in which case next group will be bad). !!! This in DC's prog does not work!!!

        rtint = []
        rampl = []
        rprops = []

        rtint.append(self.itint[n])
        rampl.append(self.iampl[n])
        rprops.append(self.iprops[n])
        ttemp = rtint[-1]
        aavtemp = rampl[-1] * rtint[-1]

        n += 1
        while n < (len(self.itint)-1):

            if self.itint[n] < tres:
                rtint[-1] += self.itint[n]
                if (rampl[-1] != 0) and (self.iampl[n] != 0):
                    aavtemp += self.iampl[n] * self.itint[n]
                    ttemp += self.itint[n]
                    rampl[-1] = aavtemp / ttemp

            else:
                if ((self.iampl[n] == 0) and (rampl[-1] == 0)):
                    rtint[-1] += self.itint[n]
                elif (rampl[-1] == self.iampl[n]) and (rampl[-1] != 0):
                    rtint[-1] += self.itint[n]
                    aavtemp += self.iampl[n] * self.itint[n]
                    ttemp += self.itint[n]
                    rampl[-1] = aavtemp / ttemp
                else:
                    rtint.append(self.itint[n])
                    rampl.append(self.iampl[n])
                    rprops.append(self.iprops[n])
                    ttemp = rtint[-1]
                    aavtemp = rampl[-1] * rtint[-1]

            if self.iprops[n] == 8:
                rprops[-1] = 8
            n += 1
        # end of while
        # TODO: check if the very last interval is closed or open.

        self.rtint = rtint
        self.rampl = rampl
        self.rprops = rprops

    def get_open_shut_periods(self):
        """
        Separate open and shut intervals from the entire record.
        
        There may be many small amplitude transitions during one opening,
        each of which will count as an individual opening, so generally
        better to look at 'open periods'.

        Look for start of a group of openings i.e. any opening that has
        defined duration (i.e. usable).  A single unusable opening in a group
        makes its length undefined so it is excluded.
        NEW VERSION -ENSURES EACH OPEN PERIOD STARTS WITH SHUT-OPEN TRANSITION
        Find start of a group (open period) -valid start must have a good shut
        time followed by a good opening -if a bad opening is found as first (or
        any later) opening then the open period is abandoned altogether, and the
        next good shut time sought as start for next open period, but for the
        purposes of identifying the nth open period, rejected ones must be counted
        as an open period even though their length is undefined.
        """

        opint = []
        oppro = []
        opamp = []
        shint = []
        shpro = []
        n = 0
        tint = 0
        prop = 0
        ampl = 0
        first = 0
        while n < len(self.rtint):
            if self.rampl[n] != 0:
                tint = tint + self.rtint[n]
                ampl = ampl + self.rampl[n] * self.rtint[n]
                if self.rprops[n] == 8: prop = 8
                avamp = ampl / tint
                n += 1
                first = 1
            else:
                shint.append(self.rtint[n])
                shpro.append(self.rprops[n])
                n += 1
                if first:
                    opamp.append(avamp)
                    avamp = 0
                    opint.append(tint)
                    tint = 0
                    oppro.append(prop)
                    prop = 0

        self.opint = opint
        self.opamp = opamp
        self.oppro = oppro
        self.shint = shint
        self.shpro = shpro

    def get_bursts(self, tcrit):
        """
        Cut entire single channel record into bursts using critical shut time
        interval (tcrit).

        Default definition of bursts:
        (1) 'Burst amplitude' defined as mean current (excluding shut
            periods) during burst;
        (2) Bursts with any 'assumed' amplitudes are INCLUDED;
        (3) Require a gap > tcrit before the 1st burst in each file;
        (4) Unusable shut time NOT a valid end of burst;
        (5) No listing of the individual bursts.

        This function generates two dictionaries- bursts and burstsopts.
        bursts- contains lists of open and shut periods in each bursts. Each
        key is burst sequence number.
        burstsopts- contains lists of burst parameters. Positions in lists
        are following: 1st position- index of first interval, 2nd position-
        index of last interval, 3rd position- burst length, 4th position-
        bad/good burst, 5th position- burst mean amplitude, 6th position-
        total open time per burst.
        """

        # Time intervals in SCN files are in ms so convert tcrit in ms too.
        tcrit = tcrit * 1000

        #defaultdef = True # True if default burst definition accepted.
        #badend = True # True if unusable shut time is valid end of burst.

        firstgapfound = False
        i = 0
        if self.rampl[0] != 0:
            firstgapfound = True

        #gap1 = 0
        bursts = {}
        burstsopts = {}

        #print("First burst starts only after gap > tcrit in each file.")
        #print("First burst starts with first good opening in each file.")
        #if badend:
        #    print("Unusable shut time treated as a valid end of burst.")
        #else:
        #    print("Unusable shut time aborts a burst.")


        while i < len(self.rtint) and not firstgapfound:
            if self.rampl[i] == 0 and self.rtint[i] > tcrit:
                firstgapfound = True
            i += 1
        #print ("First long gap found: n={0:d}; ".format(gap1) +
        #    "length = {0:.6f} ms".format(self.rtint[gap1]))

        burst = []
        burstid = 0
        burstopt = []
        newburst = True
        badburst = False
        meanamp = 0 # mean amplitude of burst
        openburst = 0 # open time per burst
        burstlen = 0 # burst len
        openinglength = 0
        openings = 0

        while i < len(self.rtint):
            if self.rampl[i] != 0:
                if newburst:
                    burst = []
                    burstopt = []
                    burstopt.append(i)
                    newburst = False
                if self.rprops[i] >= 8:
                    badburst = True
                meanamp += self.rampl[i] / self.rtint[i]
                openburst += self.rtint[i]
                burstlen += self.rtint[i]
                openinglength += self.rtint[i]
                #TODO: if bad opening: set burst bad

#                if i == len(self.rtint)-1:
#                    burstopt.append(i) # 2nd position - last interval
#                    burstopt.append(burstlen) # 3rd position- burst len
#                    if self.rprops[i] >= 8:
#                        badburst = True
#                    burstopt.append(badburst) # 4th position- bad burst
#                    burstopt.append(meanamp) # 5th position- mean ampl
#                    burstopt.append(openburst) #6thpos- opentime per burst
#
#                    burst.append(openinglength)
#                    bursts[burstid] = burst
#                    burstsopts[burstid] = burstopt
                
            else: # found gap
                if self.rtint[i] < tcrit: # TODO: and unusable gap...
                    burstlen += self.rtint[i]
                    if self.rprops[i] >= 8:
                        badburst = True
                    burst.append(openinglength)
                    openings += 1
                    burst.append(self.rtint[i])
                    openinglength = 0

                else: # gap is longer than tcrit
                    burstopt.append(i-1) # 2nd position - last interval
                    burstopt.append(burstlen) # 3rd position- burst len
                    if self.rprops[i] >= 8:
                        badburst = True
                    burstopt.append(badburst) # 4th position- bad burst
                    burstopt.append(meanamp) # 5th position- mean ampl
                    burstopt.append(openburst) #6th pos- opentime per burst
                    openings += 1
                    burstopt.append(openings) #7th pos- number of openings
                    newburst = True
                    badburst = False
                    meanamp = 0 # mean amplitude of burst
                    openburst = 0 # open time per burst
                    burstlen = 0 # burst len
                    openings = 0
                    burst.append(openinglength)
                    openinglength = 0
                    # TODO: bad/unusable gap
                    bursts[burstid] = burst
                    burstsopts[burstid] = burstopt
                    burstid += 1

            i += 1

        self.bursts = bursts
        self.burstsopts = burstsopts
        self.tcrit = tcrit * 0.001 # convert to sec

    def get_burst_length_list(self):
        n = len(self.bursts)
        blength = []
        for ind in range(n):
            blength.append(self.burstsopts[ind][2])
        return blength

    def get_openings_burst_list(self):
        n = len(self.bursts)
        openings = []
        for ind in range(n):
            openings.append(self.burstsopts[ind][6])
        return openings

    def print_bursts(self):
        print('\n#####\nList of bursts:\n')
        n = len(self.bursts)
        for ind in range(n):
            print '{0:d} '.format(ind), self.bursts[ind]
        print('\n###############\n\n')


def false_events(tres, fc, rms, amp):
    """
    Version for EKDIST/new SCAN (avamp, rms already in pA). \
    To calc false event rate (per sec) in EKDIST (from RESINT.) \
    First calc threshold as amp attained by pulse of length=tres (in ms).
    """
    u = erf(2.668 * fc * tres)
    phi = u * amp    #'threshold' (pA)
    var = (rms) ** 2    # noise variance (pA)**2
    # Calc rate from C & Sigworth eq. 9, with k=1
    frate = fc * np.exp(-(phi * phi) / (2. * var))
    return frate    # false event rate

def prepare_hist(X, tres):
    """

    """

    n = len(X)
    xmax = max(X)
    xstart = tres * 1000    # histogramm starts at

    # Defines bin width and number of bins.
    # Number of bins/decade
    if (n <= 300): nbdec = 5
    if (n > 300) and (n <= 1000): nbdec = 8
    if (n > 1000) and (n <= 3000): nbdec = 10
    if (n > 3000): nbdec = 12

    # round down minimum value, so get Xmin for distribution
    # round up maximum value, so get Xmax for distribution
    #xmin1 = int(xmin - 1)
    #xmax1 = int(xmax + 1)
    
    xend = 1. + xmax - math.fmod(xmax, 1.)    # last x value
    dx = math.exp(math.log(10.0) / float(nbdec))
    nbin = 1 + int(math.log(xend / xstart) / math.log(dx))

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

