""" Usage:
    >> python scn2txt.py scnfilename.scn
    
    or 
    
    >> python scn2txt.py scnfilename.scn tres
    
    tres : float - temporal resolution in microseconds (eg 20) 
"""

__author__="RLape"
__date__ ="$23-Apr-2015 11:54:03$"

import os
import sys
import math
from array import array
import numpy as np


class SingleChannelRecord(object):
    """
    A wrapper over a list of time intervals 
    from idealised single channel record.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('A new record initialised.')
        self.origin = None
        self.is_loaded = False
        self.record_type = None
        
        self.badopen=-1
            
    def load_SCN_file(self, infiles):
        """Load shut and open intervals from SCN file."""
        #TODO: check if infiles is valid entry: single file or a list of files 
        if isinstance(infiles, str):
            if os.path.isfile(infiles):
                infile = infiles
        elif isinstance(infiles, list):
            if os.path.isfile(infiles[0]):
                infile = infiles[0]
        #TODO: enable taking several scan files and join in a single record.
        # Just a single file could be loaded at present.
        self.header = read_header(infile, self.verbose)
        self.itint, iampl, self.iprop = read_data(
            infile, self.header)
        self.iampl = iampl.astype(float) * self.header['calfac2']
        self.origin = "Intervals loaded from SCN file: " + infile
        self.is_loaded = True
        self._tres = 0.0
        self.rtint, self.rampl, self.rprop = self.itint, self.iampl, self.iprop
        self._set_periods()
        
    def print_all_record(self):
        for i in range(len(self.itint)):
            print (i, self.itint[i], self.iampl[i], self.iprop[i])
            
    def print_resolved_intervals(self):
        print('\n#########\nList of resolved intervals:\n')
        for i in range(len(self.rtint)):
            print (i+1, self.rtint[i]*1000, self.rampl[i], self.rprop[i])
        print('\n###################\n\n')
        
    def __repr__(self):
        """String representation of SingleChannelRecord instance."""
        if not self.is_loaded:
            str_repr = "Empty record" 
        else:
            str_repr = self.origin
            str_repr += "\nTotal number of intervals = {0:d}".format(
                len(self.itint))
            str_repr += ('\nResolution for HJC calculations = ' + 
                '{0:.1f} microseconds'.format(self._tres*1e6))
            str_repr += "\nNumber of resolved intervals = {0:d}".format(
                len(self.rtint))
            str_repr += "\nNumber of time periods = {0:d}".format(
                len(self.ptint))
            str_repr += '\n\nNumber of open periods = {0:d}'.format(len(self.opint))
            str_repr += ('\nMean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.opint)*1000, np.std(self.opint)*1000))
            str_repr += ('\nRange of open periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.opint)*1000, np.max(self.opint)*1000))
            str_repr += ('\n\nNumber of shut intervals = {0:d}'.format(len(self.shint)))
            str_repr += ('\nMean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.shint)*1000, np.std(self.shint)*1000))
            str_repr += ('\nRange of shut periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.shint)*1000, np.max(self.shint)*1000))
        return str_repr
    
    def _set_resolution(self, tres=0.0):
        self._tres = tres
        self._impose_resolution()
        self._set_periods()
    def _get_resolution(self):
        return self._tres
    tres = property(_get_resolution, _set_resolution)
    
    def _impose_resolution(self):
        """
        Impose time resolution.
        First interval to start has to be resolvable, usable and preceded by
        an resolvable interval too. Otherwise its start will be defined by
        unresolvable interval and so will be unreliable.
        (1) A concantenated shut period starts with a good, resolvable
            shutting and ends when first good resolvable opening found.
            Length of concat shut period = sum of all durations before the
            resolved opening. Amplitude of concat shut period = 0.
        (2) A concantenated open period starts with a good, resolvable opening
            and ends when first good resolvable interval is found that
            has a different amplitude (either shut or open but different
            amplitude). Length of concat open period = sum of all concatenated
            durations. Amplitude of concatenated open period = weighted mean
            amplitude of all concat intervals.
        First interval in each concatenated group must be resolvable, but may
        be bad (in which case all group will be bad).
        """

        # Find negative intervals and set them unusable
        self.iprop[self.itint < 0] = 8
        # Find first resolvable and usable interval.
        n = np.intersect1d(np.where(self.itint > self._tres),
            np.where(self.iprop < 8))[0]
        
        # Initiat lists holding resolved intervals and their amplitudes and flags
        rtint, rampl, rprops = [], [], []
        # Set variables holding current interval values
        ttemp, otemp = self.itint[n], self.iprop[n]
        if (self.iampl[n] == 0):
            atemp = 0
        elif self.record_type == 'simulated':
            atemp = self.iampl[n]
        else:
            atemp = self.iampl[n] * self.itint[n]
        isopen = True if (self.iampl[n] != 0) else False
        n += 1

        # Iterate through all remaining intervals
        while n < (len(self.itint)):
            if self.itint[n] < self._tres: # interval is unresolvable

                if (len(self.itint) == n + 1) and self.iampl[n] == 0 and isopen:
                    rtint.append(ttemp)
#                    if self.record_type == 'simulated':
#                        rampl.append(atemp)
#                    else:
#                        rampl.append(atemp / ttemp)
                    rampl.append(atemp / ttemp)
                    rprops.append(otemp)
                    isopen = False
                    ttemp = self.itint[n]
                    atemp = 0
                    otemp = 8

                else:
                    ttemp += self.itint[n]
                    if self.iprop[n] >= 8: otemp = self.iprop[n]
                    if isopen: #self.iampl[n] != 0:
                        atemp += self.iampl[n] * self.itint[n]

            else:
                if (self.iampl[n] == 0): # next interval is resolvable shutting
                    if not isopen: # previous interval was shut
                        ttemp += self.itint[n]
                        if self.iprop[n] >= 8: otemp = self.iprop[n]
                    else: # previous interval was open
                        rtint.append(ttemp)
                        if self.record_type == 'simulated':
                            rampl.append(atemp)
                        else:
                            rampl.append(atemp / ttemp)
                        if (self.badopen > 0 and rtint[-1] > self.badopen):
                            rprops.append(8)
                        else:
                            rprops.append(otemp)
                        ttemp = self.itint[n]
                        otemp = self.iprop[n]
                        isopen = False
                else: # interval is resolvable opening
                    if not isopen:
                        rtint.append(ttemp)
                        rampl.append(0)
                        rprops.append(otemp)
                        ttemp, otemp = self.itint[n], self.iprop[n]
                        if self.record_type == 'simulated':
                            atemp = self.iampl[n]
                        else:
                            atemp = self.iampl[n] * self.itint[n]
                        isopen = True
                    else: # previous was open
                        if self.record_type == 'simulated':
                            ttemp += self.itint[n]
                            if self.iprop[n] >= 8: otemp = self.iprop[n]
                        elif (math.fabs((atemp / ttemp) - self.iampl[n]) <= 1.e-5):
                            ttemp += self.itint[n]
                            atemp += self.iampl[n] * self.itint[n]
                            if self.iprop[n] >= 8: otemp = self.iprop[n]
                        else:
                            rtint.append(ttemp)
                            rampl.append(atemp / ttemp)
                            if (self.badopen > 0 and rtint[-1] > self.badopen):
                                rprops.append(8)
                            else:
                                rprops.append(otemp)
                            ttemp, otemp = self.itint[n], self.iprop[n]
                            atemp = self.iampl[n] * self.itint[n]

            n += 1
        # end of while

        # add last interval
        if isopen:
            rtint.append(-1)
        else:
            rtint.append(ttemp)
        rprops.append(8)
        if isopen:
            if self.record_type == 'simulated':
                rampl.append(atemp)
            else:
                rampl.append(atemp / ttemp)
        else:
            rampl.append(0)
            
        

        self.rtint, self.rampl, self.rprop = rtint, rampl, rprops

    def _set_periods(self):
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

        pint, pamp, popt = [], [], []
        # Remove first and last intervals if shut
        if self.rampl[0] == 0:
            self.rtint = self.rtint[1:]
            self.rampl = self.rampl[1:]
            self.rprop = self.rprop[1:]
        if self.rtint[-1] < 0:
            self.rtint = self.rtint[:-1]
            self.rampl = self.rampl[:-1]
            self.rprop = self.rprop[:-1]
        while self.rampl[-1] == 0:
            self.rtint = self.rtint[:-1]
            self.rampl = self.rampl[:-1]
            self.rprop = self.rprop[:-1]

        oint, oamp, oopt = self.rtint[0], self.rampl[0] * self.rtint[0], self.rprop[0]
        n = 1
        while n < len(self.rtint):
            if self.rampl[n] != 0:
                oint += self.rtint[n]
                oamp += self.rampl[n] * self.rtint[n]
                if self.rprop[n] >= 8: oopt = 8

                if n == (len(self.rtint) - 1):
                    pamp.append(oamp/oint)
                    pint.append(oint)
                    popt.append(oopt)
            else:
                # found two consequent gaps
                if oamp == 0 and self.rampl[n] == 0 and oopt < 8:
                    pint[-1] += self.rtint[n]
                # skip bad opening
                #elif (self.badopen > 0 and oint > self.badopen) or (oopt >= 8):
                elif (oopt >= 8):
                    popt[-1] = 8
                    oint, oamp, oopt = 0.0, 0.0, 0
#                    if n != (len(self.rint) - 2):
#                        n += 1
                else: # shutting terminates good opening
                    pamp.append(oamp/oint)
                    pint.append(oint)
                    popt.append(oopt)
                    oint, oamp, oopt = 0.0, 0.0, 0
                    pamp.append(0.0)
                    pint.append(self.rtint[n])
                    popt.append(self.rprop[n])
            n += 1

        self.ptint, self.pampl, self.pprop = pint, pamp, popt
        self.opint = self.ptint[0::2]
        self.opamp = self.pampl[0::2]
        self.oppro = self.pprop[0::2]
        self.shint = self.ptint[1::2]
        self.shamp = self.pampl[1::2]
        self.shpro = self.pprop[1::2]


def read_header (fname, verbose=False):
    """
    Read SCN file header. SCN files are generated by SCAN program (DCprogs) and
    contain idealised single channel record.
    """

    # make dummy arrays to read floats, doubles and integers (LONG in C)
    floats = array ('f')
    ints = array('i')

    f = open(fname, 'rb')
    header = {}

    ints.fromfile(f,1)
    header['iscanver'] = ints.pop()
    # new scan files- version 104, 103 (simulated) and -103
    version = header['iscanver']
    if verbose: print ('version', version)

    ints.fromfile(f,1)
    ioffset = ints.pop()
    header['ioffset'] = ioffset

    ints.fromfile(f,1)
    nint = ints.pop()
    header['nint'] = nint

    header['title'] = f.read(70)
    header['date'] = f.read(11)

    if version == -103 or version == 103:
        header['tapeID'] = f.read(24)
        ints.fromfile(f,1)
        header['ipatch'] = ints.pop()
        floats.fromfile(f,1)
        header['Emem'] = floats.pop()
        ints.fromfile(f,1)
        header['unknown1'] = ints.pop()
        floats.fromfile(f,1)
        header['avamp'] = floats.pop()
        floats.fromfile(f,1)
        header['rms'] = floats.pop()
        floats.fromfile(f,1)
        header['ffilt'] = floats.pop()
        floats.fromfile(f,1)
        calfac2 = floats.pop()
        header['calfac2'] = calfac2
        floats.fromfile(f,1)
        header['treso'] = floats.pop()
        floats.fromfile(f,1)
        header['tresg'] = floats.pop()
        if version == 103:
            header['type'] = 'simulated'

        f.close()
        return header

    header['type'] = 'experiment'
    header['defname'] = f.read(6)
    header['tapeID'] = f.read(24)
    ints.fromfile(f,1)
    header['ipatch'] = ints.pop()
    ints.fromfile(f,1)
    header['npatch'] = ints.pop()
    floats.fromfile(f,1)
    header['Emem'] = floats.pop()
    floats.fromfile(f,1)
    header['temper'] = floats.pop()
    header['adcfil'] = f.read(30)
    header['qfile1'] = f.read(35)

    # logical; true if data from CJUMP file
    ints.fromfile(f,1)
    header['cjump'] = ints.pop()

    ints.fromfile(f,1)
    header['nfits'] = ints.pop()

    ints.fromfile(f,1)
    header['ntmax'] = ints.pop()

    ints.fromfile(f,1)
    header['nfmax'] = ints.pop()

    # Number of data points read into memory at each disk read -bigger
    # the better (max depends on how much RAM you have).
    # nbuf=131072		!=1024*128
    ints.fromfile(f,1)
    header['nbuf'] = ints.pop()

    # Number of extra points read in, at each end of data section to
    # allow display of transitions on section boundaries; 2048 is OK usually.
    ints.fromfile(f,1)
    header['novlap'] = ints.pop()

    # Sample rate (Hz)
    floats.fromfile(f,1)
    header['srate'] = floats.pop()

    # finter = microsec between data points; finter=1.e6/srate
    floats.fromfile(f,1)
    header['finter'] = floats.pop()

    # TSECT=time (microsec) from first point of one section to first point of next
    # tsect=float(nbuf)*finter
    floats.fromfile(f,1)
    header['tsect'] = floats.pop()

    # The first data point in data file, idata(1) starts at byte (record #) ioff+1
    ints.fromfile(f,1)
    header['ioff'] = ints.pop()

    ints.fromfile(f,1)
    header['ndat'] = ints.pop()

    # calc nsec etc here, in case default nbuf altered
    # if(ndat.lt.nbuf) nbuf=ndat    !allocate smaller array
    # nsec= 1 + (ndat-1)/nbuf  !number of sections
    ints.fromfile(f,1)
    header['nsec'] = ints.pop()

    # nrlast=ndat - (nsec-1)*nbuf  !number of idata in last section
    ints.fromfile(f,1)
    header['nrlast'] = ints.pop()

    floats.fromfile(f,1)
    header['avtot'] = floats.pop()

    ints.fromfile(f,1)
    header['navamp'] = ints.pop()

    floats.fromfile(f,1)
    avamp = floats.pop()
    header['avamp'] = avamp

    floats.fromfile(f,1)
    rms = floats.pop()
    header['rms'] = rms

    # Data will be written to disk at (approx) every nth transition, so
    # analysis can be restarted by using the ''restart'' option when SCAN reentered.
    ints.fromfile(f,1)
    header['nwrit'] = ints.pop()

    # nwsav=0		!used for auto disc write
    ints.fromfile(f,1)
    header['nwsav'] = ints.pop()

    # logical
    ints.fromfile(f,1)
    header['newpar'] = ints.pop()

    # logical
    ints.fromfile(f,1)
    header['opendown'] = ints.pop()

    # logical; Invert trace (openings must be downwards)
    ints.fromfile(f,1)
    header['invert'] = ints.pop()

    # logical; usepots=.false.
    ints.fromfile(f,1)
    header['usepots'] = ints.pop()

    # in SCAN: Display only (no step-response function)
    ints.fromfile(f,1)
    header['disp'] = ints.pop()

    # if(iscrit.eq.1): Percentage of full amplitude for critical level
    # (Scrit) beyond which transition is deemed to occur.
    # if(iscrit.eq.2): Multiple of RMS noise to define critical level
    # (Scrit) beyond which transition is deemed to occur.
    # if(iscrit.eq.1): smult=0.14 !scrit=0.14*avamp
    # if(iscrit.eq.2): smult=5. scrit=5.0*rms
    floats.fromfile(f,1)
    header['smult'] = floats.pop()

    ints.fromfile(f,1)
    header['scrit'] = ints.pop()

    ints.fromfile(f,1)
    header['vary'] = ints.pop()

    # Number of consecutive points beyond Scrit for a transition to be
    # deemed to have occurred. Default: ntrig=2
    ints.fromfile(f,1)
    header['ntrig'] = ints.pop()

    # navtest=ntrig-1 ; if(navtest.le.0) navtest=1
    # navtest=number averaged before average curlev is used, rather than
    # input curlev in FINDTRANS (NB must be less than ntrig, or, for
    # example, if input baseline is not close to current baseline
    # (ie baseline has drifted since last time) then will get a 'trigger'
    # straight away!
    ints.fromfile(f,1)
    header['navtest'] = ints.pop()

    # Trace will be amplified by this factor before display (but better to
    # amplify correctly BEFORE sampling). DGAIN=1.0
    floats.fromfile(f,1)
    header['dgain'] = floats.pop()

    # IBOFF=0		!BASELINE OFFSET FOR DISPLAY (ADC)
    ints.fromfile(f,1)
    header['iboff'] = ints.pop()

    # Factor by which trace is expanded when ''expand'' is first hit.
    # expfac=2.
    floats.fromfile(f,1)
    header['expfac'] = floats.pop()

    # Position of baseline on screen is offset to this level after initial
    # ''get piece of baseline on screen'' is completed.
    # bdisp=0.75 if openings downwards; bdisp=0.25 if openings upwards
    floats.fromfile(f,1)
    header['bdisp'] = floats.pop()

    ints.fromfile(f,1)
    header['ibflag'] = ints.pop()

    # Auto-fit to avoid sublevels if possible. In case of doubt fit brief
    # open-shut-open rather than fitting a sublevel.
    ints.fromfile(f,1)
    header['iautosub'] = ints.pop()

    # When opening crosses the red trigger line display stops with the
    # opening transition at this point on the x-axis of display.
    # xtrig=0.2: trigger at 20% of X axis on screen
    floats.fromfile(f,1)
    header['xtrig'] = floats.pop()

    # ndev='C:'; disk partition for Windows
    header['ndev'] = f.read(2)
    header['cdate'] = f.read(11)
    header['adctime'] = f.read(8)

    ints.fromfile(f, 1)
    header['nsetup'] = ints.pop()

    header['filtfile'] = f.read(20)

    # Low pass filter (Hz, -3dB)
    # later needs to be converted to kHz
    floats.fromfile(f, 1)
    ffilt = floats.pop()
    header['ffilt'] = ffilt

    # npfilt=number of points to jump forward after a transition, to start
    # search for next transition
    # npfilt1= number of data points for filter to go from 1% to 99%
    # npfilt1=ifixr((tf99-tf1)/finter)
    # npfilt=ifixr(float(npfilt1)*facjump)
    ints.fromfile(f, 1)
    header['npfilt'] = ints.pop()

    # sfac1=(yd2-yd1)/65536.
    # sfac1=sfac1*dgain			!true scal fac for ADC to pixel units
    floats.fromfile(f, 1)
    header['sfac1'] = floats.pop()

    # nscale=1 + ifix(alog(4096./(yd2-yd1))/alog(2.))
    # sfac2=sfac1*float(2**nscale)	!Converts ADC units to intermed units
    floats.fromfile(f, 1)
    header['sfac2'] = floats.pop()

    # sfac3=1.0/float(2**nscale) 	!converts intermed to pixel units
    floats.fromfile(f, 1)
    header['sfac3'] = floats.pop()

    ints.fromfile(f, 1)
    header['nscale'] = ints.pop()

    # Calibration factor (pA per ADC unit)
    floats.fromfile(f, 1)
    header['calfac'] = floats.pop()

    # calfac1=calfac/sfac1		!converts pixel display units to pA
    floats.fromfile(f, 1)
    header['calfac1'] = floats.pop()

    # calfac2=calfac/sfac2		!converts intermed units to pA
    floats.fromfile(f, 1)
    calfac2 = floats.pop()
    header['calfac2'] = calfac2

    f.close() #    close the file
    return header

def read_data(fname, header):
    """
    Read idealised data- intervals, amplitudes, flags- rom SCN file.
    Data=
    real*4 tint(1...nint) 	 4nint bytes
    integer*2 iampl(1..nint)   2nint bytes
    integer*1 iprops(1..nint)  nint  bytes
    Total storage needed = 7 * nint bytes
    integer*1 iprops(i) holds properties of ith duration and amplitude
    (integer*1 has range -128 to +127 (bit 7 set gives -128; can use bits 0-6)
    0 = all OK;
    1 = amplitude dubious = bit 0;
    2 = amplitude fixed = bit 1;
    4 = amplitude of opening constrained (see fixamp) = bit 2;
    8 = duration unusable = bit 3; etc
    and keep sum of values of more than one property is true.
    """

    tint = array ('f') # 4 byte float
    iampl = array ('h') # 2 byte integer
    iprops = array('b') # 1 byte integer

    f=open(fname, 'rb')
    f.seek(header['ioffset']-1)
    tint.fromfile(f, header['nint'])
    iampl.fromfile(f, header['nint'])
    iprops.fromfile(f, header['nint'])
    f.close()
     
    if header['iscanver'] > 0:
        gapnotfound = True
        while gapnotfound:
            if iampl[-1] == 0:
                gapnotfound = False
                iprops[-1] = 8
            else:
                tint.pop()
                iampl.pop()
                iprops.pop()
        
    return np.array(tint)*0.001, np.array(iampl), np.array(iprops)

def write_to_txt(foutname, itint, iampl, iprops):
    fout = open(foutname, 'w')
    for i in range(len(itint)):
        fout.write('{0:.16e}\t{1:.3f}\t{2:d}\n'.
            format(itint[i], iampl[i], iprops[i]))
    fout.close()
    return foutname


if __name__ == "__main__":

    if len(sys.argv) > 1:

        fname = sys.argv[1]
        print ('Converting file: ', fname)
        ftxtname = fname[:-4] + '.txt'
        rec = SingleChannelRecord()
        rec.load_SCN_file([fname])

        if len(sys.argv) > 2:
            tres = float(sys.argv[2])
            rec.tres = tres * 1e-6
            print("Imposed temporal resolution of {0:.3f} microseconds".format(tres))
            write_to_txt(ftxtname, rec.ptint, rec.pampl, rec.pprop)
        else:
            write_to_txt(ftxtname, rec.itint, rec.iampl, rec.iprop)

        print ('Saved TXT file: ', ftxtname)
        print ('First column in text file: intervals in seconds.')
        print ('Second column in text file: amplitudes in pA.')
        print ('Third column in text file: flags; if flag>=8 then interval is bad or unusable.')
        print ('Done!')
    else:
        print ("No file was converted.")
        print ("Please, use this script this way: 'python scn2txt.py file_to_convert_name.scn'")




