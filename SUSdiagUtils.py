##############################################################
# Collection of functions used in suspension diagonalization #
##############################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import nds2
import timeit, tqdm, os, datetime
import yaml
import h5py
import gpstime, re
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties
import tabulate
import sympy
import logging
import socket 

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format="%(levelname)s \n%(message)s")


# Directory setup
globFigDir = 'Figures/'
globDataDir = 'Data/'
if globFigDir.strip('/') not in os.listdir():
    #print('Figures directory not found, making it...')
    logging.debug('Figures directory not found, making it...')
    os.mkdir(globFigDir)
if globDataDir.strip('/') not in os.listdir():
    #print('Data directory not found, making it...')
    logging.debug('Data directory not found, making it...')
    os.mkdir(globDataDir)

# Matplotlib setup
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize']  = 'large'
plt.rcParams['axes.labelsize']  = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.formatter.limits'] = [-2,2]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['text.usetex'] = False
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['figure.figsize'] = [12,9]
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.subplot.left'] = 0.07
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.top'] = 0.92
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'black'


def importParams(paramFile):
    '''
    Load in a set of parameters.
    '''
    with open(paramFile,'r') as f:
        params = yaml.safe_load(f)
    return(params)

def dlData(paramFile):
    '''
    Function that downloads data, and saves it to a HDF5 file
    after downsampling to rate specified in parameter file.
    '''
    # Load the parameter file
    par = importParams(paramFile)
    # Check if the data file exists alread
    if par['optic']+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5' in os.listdir(dataDir):
        #print('Data file exists already...')
        logging.debug('Data file exists already...')
        return
    else:
        # Define the NDS connection parameters
        if 'rossa' in socket.gethostname():
            ndsServer = 'fb'
            ndsPort = 8088
        else:
            ndsServer = par['ndsServer']
            ndsPort = par['ndsPort']
        ndsParams = nds2.parameters()
        ndsParams.set('GAP_HANDLER','STATIC_HANDLER_ZERO')
        # Define the downsampling params
        dsFactor = par['dsFactor']
        # Define the channels 
        optic = par['optic']
        chans = []
        for ii in ['LL','LR','UL', 'UR','SIDE']:
            chans.append('C1:SUS-'+optic+'_SENSOR_'+ii)
        # Open the NDS connection
        try:
            conn = nds2.connection(ndsServer, ndsPort)
            #print('NDS2 connection opened...')
            logging.debug('NDS2 connection opened...')
        except:
            #print('Unable to open NDS connection')
            logging.debug('Unable to open NDS connection')
        # Make the HDF5 file to save the fft
        fil = h5py.File(dataDir+optic+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5','w')
        # Get the data
        for ii, jj in tqdm.tqdm(zip(chans, ['LL','LR','UL', 'UR','SIDE']), total=len(chans)):
            cc = fil.create_group(f'tSeries_{jj}')
            for mm, tt in enumerate(par['tStart']):
                tStart = tt
                tEnd = tStart + par['tDur'] 
                #print('Fetching {} from {} to {} ...'.format(ii,tStart, tEnd))
                logging.debug(f'Fetching {ii} from {tStart} to {tEnd} ...')
                #dat = conn.fetch(int(tStart), int(tEnd), [ii])
                dat = nds2.fetch([ii], int(tStart), int(tEnd), params=ndsParams)
                # Do the detrending and downsampling
                yy = sig.detrend(dat[0].data[:])
                yy = sig.decimate(yy, dsFactor, ftype='fir')
                # Save this to file
                tSer = cc.create_dataset(f'{mm}', data=yy)
                tSer.attrs['fs'] = dat[0].sample_rate / dsFactor
        fil.close()
    
def highResSpectra(paramFile):
    '''
    Function that opens a data file, computes a high res spectral density
    '''
    # Load the parameter file
    par = importParams(paramFile)
    optic = par['optic']
    # Open the HDF5 file with the time series data
    fil = h5py.File(dataDir+optic+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5','a')
    # FFT params
    tFFT = par['tFFT']
    # Get the data
    for ii in tqdm.tqdm(fil.keys()):
        if 'tSeries_' in ii:
            for kk in tqdm.tqdm(fil[ii].keys()):
                # Setup the FFT parameters
                fs = fil[ii][kk].attrs['fs']
                nFFT = int(tFFT * fs)
                win = sig.get_window(('tukey',0.25),nFFT)
                # Do the FFT
                #print('Welch-ing {} ...'.format(re.sub('tSeries_','',ii)))
                logging.debug('Welch-ing {} ...'.format(re.sub('tSeries_','',ii)))
                ff, Pxx = sig.welch(fil[ii][kk][:], fs=fs, window=win, scaling='density')
                if kk=='0':
                    pows = Pxx
                else:
                    pows = np.vstack((pows,Pxx))
            # Compute the median
            Pxx_median = np.median(pows, axis=0)
            # Save this to file
            if 'ff_'+re.sub('tSeries_','',ii) in fil.keys():
                del fil['ff_'+re.sub('tSeries_','',ii)]
                del fil['P_'+re.sub('tSeries_','',ii)]
            fil['ff_'+re.sub('tSeries_','',ii)] = ff
            fil['P_'+re.sub('tSeries_','',ii)] = np.sqrt(Pxx) * np.sqrt(np.log(2)) # Median averaging of the periodograms
    fil.close()
    return

def plotSpectra(paramFile, fig, ax, zoom=True, brMode=False, saveFig=True):
    '''
    Opens the HDF5 file containing the suspension spectra and plot it
    '''
    # Load the parameter file
    par = importParams(paramFile)
    fil = h5py.File(dataDir+par['optic']+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5','r')
    for ii in ['UL', 'UR', 'LL', 'LR', 'SIDE']:
        ax.semilogy(fil['ff_'+ii], fil['P_'+ii], '.', label=ii, rasterized=True)
    ax.grid(True, which='both')
    ax.legend(loc='best')
    ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Sensor voltage ASD [$\mu m/\sqrt{\mathrm{Hz}}$]')
    ax.set_ylabel('Sensor voltage ASD [cts/$\sqrt{\mathrm{Hz}}$]')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    fig.suptitle(par['optic']+' Sensor voltage spectra from {} for {} seconds'.format(par['tStart'][0], par['tDur']))
    if zoom:
        if brMode:
            ax.set_xlim([15,25])
            ax.set_ylim([1e-4,1])
        else:
            ax.set_xlim([0.5,1.2])
            ax.set_ylim([1e-2, 300])
    if saveFig:
        fig.savefig(figDir+par['optic']+'_sensorSpectra.pdf', bbox_inches='tight')
    return

def plotTimeSeries(paramFile, fig, ax, saveFig=True):
    '''
    Tile plot the time series (downsampled)
    used to make the spectra
    '''
    # Load the parameter file
    par = importParams(paramFile)
    fil = h5py.File(dataDir+par['optic']+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5','r')
    #Some fancy boxed text
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    for ii, jj in enumerate([ v for v in fil.keys() if 'tSer' in v]):
        if 'tSeries_' in jj and 'SIDE' not in jj:
            yy = fil[jj]['0'][:]
            tt = np.linspace(0, (len(yy)-1)/fil[jj]['0'].attrs['fs'], len(yy))/1000
            ax[int(np.floor(ii/3)), int(ii%3)].plot(tt, yy, rasterized=True, alpha=0.5, label=re.sub('tSeries_','',jj))
            ax[int(np.floor(ii/3)), int(ii%3)].text(0.75,0.05,re.sub('tSeries_','',jj), transform=ax[int(np.floor(ii/3)), int(ii%3)].transAxes, bbox=props, fontsize=22,fontweight='bold')
        ax[0,2].axis('off')
        ax[0,2].text(0.5,0.5,'Detrended, downsampled, \n free-swinging data \n for {}'.format(par['optic']))
    ax[1,2].plot(tt, fil['tSeries_SIDE']['0'][:], rasterized=True, alpha=0.5, label=re.sub('tSeries_','',jj))
    ax[1,2].text(0.75,0.05,'SIDE', transform=ax[1,2].transAxes, bbox=props, fontsize=22,fontweight='bold')
    fil.close()
    ax[1,0].set_xlabel('Time [ksec]')
    ax[1,1].set_xlabel('Time [ksec]')
    ax[1,2].set_xlabel('Time [ksec]')
    ax[0,0].set_ylabel('Amp [cts]')
    ax[1,0].set_ylabel('Amp [cts]')
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%2d'))
    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%2d'))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle('Time series data for free-swinging {}'.format(par['optic']))
    if saveFig:
        fig.savefig(figDir+par['optic']+'_sensors_timeDomain'+'.pdf', bbox_inches='tight')


# Some helper functions
def OSEM2Eul(paramFile, mtrx, fftParams, nullStream=False):
    '''
    Compute the DoF streams from the OSEM basis.
    Option to also get the nullstream (pringle/butterfly mode)
    
    Parameters:
    -----------
    paramFile: str
        Path to parameter file
    nullStream: bool
        Whether to calculate the nullstream or not
    fftParams: dict
        A dictionary of fft parameters
        
    Returns:
    ---------
    ff: array_like
        Frequency vector for ASD
    Eul: array_like
        A matrix of the spectra in the DoF basis, whose ROWS are the DoFs
    fs: float
        Sampling frequency of the timeseries data [Hz]
    '''
    # Load the parameter file
    par = importParams(paramFile)
    fil = h5py.File(dataDir+par['optic']+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5','r')
    if nullStream:
        mtrx = np.vstack((mtrx,np.array([1,-1,-1,1,0])))
    #OSEMs = np.array([fil['P_UL'][:], fil['P_UR'][:], fil['P_LL'][:], fil['P_LR'][:], fil['P_SIDE'][:]])
    for kk in fil['tSeries_LL'].keys():
        OSEMs = np.array([fil['tSeries_UL'][kk][:], fil['tSeries_UR'][kk][:], fil['tSeries_LL'][kk][:], fil['tSeries_LR'][kk][:], fil['tSeries_SIDE'][kk][:]])
        #ff = fil['ff_UL'][:]
        fs = fil['tSeries_UL'][kk].attrs['fs']
        
        Eul_T = np.dot(mtrx, OSEMs)
        nFFT = int(fftParams['tFFT'] * fs)
        for ii in range(len(Eul_T)):
            ff, Pxx = sig.welch(Eul_T[ii,:], fs=fs, nperseg=nFFT, window=sig.get_window(fftParams['window'],nFFT))
            if ii==0:
                pspec = Pxx
            else:
                pspec = np.vstack((pspec, Pxx))
        pspec = np.broadcast_to(pspec, (1, pspec.shape[0], pspec.shape[1]))
        if kk=='0':
            pows = pspec
        else:
            pows = np.vstack((pows, pspec))
    Pxx_med = np.median(pows, axis=0)
    fil.close()
    Eul = np.sqrt(Pxx_med) * np.sqrt(np.log(2))
    return(ff, Eul, fs)

def lor(freq, cent, wid, amp):
    '''
    Computes a lorentzian.
    '''
    num = 0.5*wid
    den = (freq - cent)**2 + (0.5*wid)**2
    return(amp*num / den / np.pi)


def fitSpectra(paramFile, guessFile, fftParams, fig, ax, mtrx=np.array([[1,1,1,1,0],[1,1,-1,-1,0],[1,-1,1,-1,0],[0,0,0,0,1]]), nullStream=False):
    '''
    Function that takes OSEM sensor spectra, makes DOF streams from them, finds peaks, and fits Lorentzians to them.
    Scipy's curve_fit is used to fit the Lorentzian.
    
    Parameters:
    ------------
    paramFile: str
        Path to parameter file
    guessFile: str
        Path to a file containing initial guesses for everything
    fftParams: dict
        A dictionary of fft parameters.
    fig: matplotlib figure object
        Matplotlib figure object on which to plot stuff
    ax: matplotlib axis object
        Axis onto which to make the plot
    mtrx: array_like
        Matrix with which to convert individual sensor streams to DoF basis
    nullStream: bool
        Whether to include butterfly mode in the plot or not. Defaults to False.

    Returns:
    ---------
    fitDict: dict
        Dictionary of outcomes from the fitting
    '''
    # Start by checking the size of the input matrix
    if mtrx.shape != (4,5):
        #print('Dimensionality of the input matrix is NOT [4,5] - please try again!')
        logging.debug('Dimensionality of the input matrix is NOT [4,5] - please try again!')
        return
    
    # Initialize a dictionary object
    fitDict = {}
    # DoFs
    DoFs = ['POS','PIT','YAW','SIDE']
    for ii in DoFs:
        fitDict[ii] = {}
    # Start by making the DoFs
    ff, Eul, fs = OSEM2Eul(paramFile, mtrx, fftParams=fftParams, nullStream=nullStream)
    # Load the parameter file for the fitting
    par = importParams(paramFile)
    # Get the initial guesses
    inits = importParams(guessFile)
    inits = inits[par['optic']]
    tbl = []
    # Find the locations of the peaks
    for ii in range(len(Eul)-1):
        ax.semilogy(ff, Eul[ii,:], '.', label=DoFs[ii], rasterized=True)
        #print('Finding peaks for {}....'.format(DoFs[ii]))
        logging.debug('Finding peaks for {}....'.format(DoFs[ii]))
        subsp = [np.argmin(np.abs(ff - inits['f0'][ii] + inits['df'])), np.argmin(np.abs(ff - inits['f0'][ii] - inits['df'] ))]
        peak_ind, props = sig.find_peaks(Eul[ii,subsp[0]:subsp[1]], height=10) # Look for peaks at least 1 um/rtHz tall
        # Add the offset to refer back to the original frequency vector 
        peak_ind += subsp[0]
        if len(peak_ind)==1:
            #print('Found peaks in {}. Now fitting...'.format(DoFs[ii]))
            logging.debug('Found peaks in {}. Now fitting...'.format(DoFs[ii]))
            #ff_trunc = ff[int(peak_ind-15):int(peak_ind+15)]
            #dat_trunc = Eul[ii,int(peak_ind-15):int(peak_ind+15)]
            deltaF = inits['Df']
            nPts = deltaF / (ff[1] - ff[0])
            ff_trunc = ff[int(peak_ind-nPts):int(peak_ind+nPts)]
            dat_trunc = Eul[ii,int(peak_ind-nPts):int(peak_ind+nPts)]
            p0 = [inits['f0'][ii], inits['gam'][ii], inits['A'][ii]]
            try:
                popt, pcov = curve_fit(lor, ff_trunc, dat_trunc, p0=p0, bounds=([0, 0, 0],[np.inf, 0.05, np.inf]), ftol=1e-3, maxfev=int(1e6))
                #popt, pcov = curve_fit(lor, ff_trunc, dat_trunc, p0=p0, ftol=1e-3, maxfev=int(1e6))
                logging.debug('Fitted {}'.format(DoFs[ii]))
                fitDict[DoFs[ii]]['pkLoc'] = peak_ind
                fitDict[DoFs[ii]]['pkFreq'] = ff[peak_ind]
                fitDict[DoFs[ii]]['pOpt'] = popt
                fitDict[DoFs[ii]]['pCov'] = pcov
                # Make the plot
                ff_ext = ff[int(peak_ind-100):int(peak_ind+100)]
                dat_ext = Eul[ii,int(peak_ind-100):int(peak_ind+100)]
                #ax.semilogy(ff, Eul[ii,:], '.', label=DoFs[ii], rasterized=True)
                ax.semilogy(ff_ext, lor(ff_ext, *popt), alpha=0.7, color=ax.lines[-1].get_color())
                q = popt[0] / popt[1]
                tbl.append([DoFs[ii], '{} +/- {}'.format(round(popt[0],4), roundTo1(np.sqrt(pcov[0,0]))),  '{}'.format(round(q,3)) ])
            except Exception as e:
                logging.debug(e)
                continue
        else:
            ax.semilogy(ff, Eul[ii,:], '.', label=DoFs[ii], rasterized=True)
            logging.debug('WARNING: {} peaks found between {} Hz and {} Hz. Skipping {}'.format(len(peak_ind), ff[subsp[0]], ff[subsp[1]], DoFs[ii]))
            continue
    if nullStream:
        ax.semilogy(ff, Eul[-1,:], label='NULL', rasterized=True)
    ax.grid(True, which='both')
    ax.legend(loc='best')
    ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Sensor voltage ASD [$\mu m/\sqrt{\mathrm{Hz}}$]')
    ax.set_ylabel('Sensor voltage ASD [cts/$\sqrt{\mathrm{Hz}}$]')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlim([0.5,1.2])
    ax.set_ylim([1e-2, 300])
    fig.suptitle('Peak fitting using the naive input matrix for {}'.format(par['optic']))
    # Add the table to the plot
    plotTbl = ax.table(cellText=tbl, fontsize=14, colLabels=['DoF', '$f_0$','Q'], loc='upper left', colWidths=[0.05,0.1,0.05], rowLoc='center', colLoc='center')
    for (row, col), cell in plotTbl.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='extra bold'))
    fig.savefig(figDir+par['optic']+'_pkFitNaive'+'.pdf', bbox_inches='tight')
    return(fitDict)      

def cplxTF(paramFile, fftParams, fig, ax, fitDict, mtrx=np.array([[1,1,1,1,0],[1,1,-1,-1,0],[1,-1,1,-1,0],[0,0,0,0,1]]), nullStream=False):
    '''
    Function that evaluates the COMPLEX coil-to-coil transfer function matrices.
    
    Parameters:
    ------------
    paramFile: str
        Path to parameter file
    fftParams: dict
        A dictionary of fft parameters.
    fig: matplotlib figure object
        Matplotlib figure object on which to make the plot
    ax: matplotlib axis object
        Axis onto which to make the plot
    fitDict: dictionary
        Dictionary output of the fitSpectra function, used for zooming in the plots around the peaks.
    mtrx: array_like
        Matrix with which to convert individual sensor streams to DoF basis. Defaults to the "Naive" matrix
    nullStream: Bool
        Include Nullstream or not.

    Returns:
    ---------
    TF: array_like
        Complex matrix of TFs
    '''
    # Load the data file
    par = importParams(paramFile)
    try:
        fname = dataDir+par['optic']+'_sensors_'+re.sub(' ', '_', str(par['tStart'][0]))+'.hdf5'
        print(f'Loading {fname}')
        fil = h5py.File(fname,'r')
        print('Loaded data file...')
        for kk in fil['tSeries_LL'].keys():
            if kk=='0':
                tmp = np.array([fil['tSeries_UL'][kk][:], fil['tSeries_UR'][kk][:], fil['tSeries_LL'][kk][:], fil['tSeries_LR'][kk][:], fil['tSeries_SIDE'][kk][:]])
                OSEMs = np.broadcast_to(tmp, (1, tmp.shape[0], tmp.shape[1]))
            else:
                OSEMs = np.vstack((OSEMs, np.broadcast_to(tmp, (1, tmp.shape[0], tmp.shape[1]))))
        fs = fil['tSeries_UL']['0'].attrs['fs'] # Downsampled data
        fil.close()
    except Exception as e:
        print('Data file doesnt exist!')
        logging.debug(e)
        logging.debug('Data file doesnt exist!')
        return
    
    # Compute the CSDs and PSDs from everything to UL
    nFFT = int(fftParams['tFFT'] * fs)
    for kk in range(OSEMs.shape[0]):
        for ii in range(OSEMs.shape[1]):
            ff, Pyx = sig.csd(OSEMs[kk, ii,:],OSEMs[kk,0,:], fs=fs, nperseg=nFFT, window=sig.get_window(fftParams['window'],nFFT))
            ff, Pxx = sig.welch(OSEMs[kk, 0,:], fs=fs, nperseg=nFFT, window=sig.get_window(fftParams['window'],nFFT))
            if ii==0:
                TF = Pyx / Pxx
            else:
                TF = np.vstack((TF, Pyx/Pxx))
        TF_stacked = np.broadcast_to(TF, (1, TF.shape[0], TF.shape[1]))
        if kk==0:
            TFF = TF_stacked
        else:
            TFF = np.vstack((TFF, TF_stacked))
    TF = np.mean(TFF, axis=0)
    # Make the plot
    labs = ['UL-UL','UR-UL','LL-UL','LR-UL','SD-UL']
    # DoFs
    DoFs = ['POS','PIT','YAW','SIDE']
    for ii in range(len(TF)):
        for plotInd in range(len(DoFs)):
            ax[0, plotInd].semilogy(ff, np.abs(TF[ii,:]), label=labs[ii], rasterized=True)
            ax[1, plotInd].plot(ff, np.angle(TF[ii,:], deg=True), alpha=0.7, linestyle='--', rasterized=True)
    # Add the Peaks
    # Start by making the DoFs
    ff, Eul, fs = OSEM2Eul(paramFile, mtrx, fftParams=fftParams, nullStream=nullStream)
    # Do the FFT
    nFFT = int(fftParams['tFFT'] * fs)
    for ii in range(len(Eul)-1):
        for plotInd in range(len(DoFs)):
            ax[2, plotInd].semilogy(ff, Eul[ii,:], label=DoFs[ii], rasterized=True)
    ax[1,0].set_ylim([-185,185])
    ax[2,0].set_ylim([1e-1,300])
    ax[1,0].set_yticks(np.linspace(-180,180,9))
    ax[1,0].set_ylabel('Phase [deg]')
    ax[0,0].set_ylabel('Magnitude [abs]')
    ax[2,0].set_ylabel('ASD [cts/$\sqrt{\mathrm{Hz}}$]')
    ax[0,0].legend(loc='best')
    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter("%2d"))
    ax[2,0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax[2,0].xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    for bb in ax:
        for aa in bb:
            aa.grid(True,which='both')
    for plotInd in range(len(DoFs)):
        pkCenter = fitDict[DoFs[plotInd]]['pkFreq']
        ax[2, plotInd].set_xlim([pkCenter-0.01, pkCenter+0.01])
        ax[0, plotInd].vlines(pkCenter, 0.1,10, linewidth=0.5, color='xkcd:electric pink')
        ax[1, plotInd].vlines(pkCenter, -180,180, linewidth=0.5, color='xkcd:electric pink')
        ax[0, plotInd].set_title(DoFs[plotInd])
    fig.text(0.5, 0.04, 'Frequency [Hz]', ha='center', fontsize=32) # Common x label
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle('Complex TF from all coils to UL for {}'.format(par['optic']))
    fig.savefig(figDir+par['optic']+'_cplxTF'+'.pdf', bbox_inches='tight')
    return(ff, TF)

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sympy.Number)})

def calcSensMat(paramFile, fftParams, fitDict, TF, ff_TF, fig, ax, naiveMat=np.array([[1,1,1,1,0],[1,1,-1,-1,0],[1,-1,1,-1,0],[0,0,0,0,1]])):
    '''
    Function to use complex TF info from free-swinging optic to calculate the
    sensing matrix that leads to good decoupling of the eigenmodes.
    
    Parameters:
    -----------
    paramFile: str
        Parameter file
    fftParams: dict
        Dictionary of FFT parameters
    fitDict: dict
        Dictionary output from fitSpectra
    TF: array_like
        Matrix of complex TFs, output from cplxTF
    ff_TF: array_like
        Frequency vector to go with TF
    fig: matplotlib figure object
        For plotting the pre/post diagonalization spectra
    ax: matplotlib axes
        For plotting the pre/post diagonalization spectra
    naiveMat: array_like
        The guess for what the input matrix should be
    
    Returns:
    --------
    invMat: array_like
        Calculated input matrix from free-swinging data, un-normalized
    phases:array_like
        Array of phases, to check if the sign of TFs makes sense
    realSenMat: array_like
        Real part of the inferred matrix from complex TF data.
        Butterfly mode is included.
    sensMat: array_like
        Complex TFs from OSEM to DoF. No butterfly mode.
    cond: array_like
        Condition number of the (inverse of) the matrix that the 
        matrix that naiveMat has to be multiplied by, to get the
        calculated sensing matrix. Should be close to 1.
    '''
    par = importParams(paramFile)
    DoFs = ['POS', 'PIT', 'YAW', 'SIDE']
    OSEMs = ['UL','UR','LL','LR','SIDE']
    sensMat = np.zeros((len(DoFs),len(OSEMs)), dtype='complex') # 4 DoFs, 5 columns
    phases = np.zeros((len(DoFs),len(OSEMs))) # 4 DoFs, 5 columns
    for ll, ii in enumerate(DoFs):
        pkF = fitDict[ii]['pkFreq']
        # Find the closest element in the complex TF
        kk = np.argmin(np.abs(ff_TF - pkF)) # This corresponds to a column index in the TF array
        for jj in range(len(TF)):
            avgSet = TF[jj,int(kk-1):int(kk+1)]
            # Take care of the complex number division
            sensMat[ll, jj] = np.sum(np.real(avgSet))/len(avgSet) + 1j * np.sum(np.imag(avgSet))/len(avgSet)
            phases[ll, jj] = np.angle(sensMat[ll,jj], deg=True)
            # At the same time, check that ratio of imag to real part is < 0.1
            if np.abs(np.imag(sensMat[ll,jj])/np.real(sensMat[ll,jj])) > 0.1:
                #print('WARNING: The imaginary part of the {} OSEM to {} is large!'.format(OSEMs[jj], ii))
                logging.debug('WARNING: The imaginary part of the {} OSEM to {} is large!'.format(OSEMs[jj], ii))
    realSenMat = np.real(sensMat)
    realSenMat =  np.vstack((realSenMat, np.array([1, -1, -1, 1, 0]))) # Add the butterfly mode
    invMat = np.linalg.inv(realSenMat)
    invMatNorm = np.copy(invMat)
    # Normalize by the largest element in the row
    for ii in range(len(invMat)):
        invMatNorm[ii, :] = invMat[ii,:] / np.mean(np.abs(invMat[ii,:]))
    
    # For a measure of badness, check the condition number
    naiveMat = np.array([[1,1,1,1,0],[1,1,-1,-1,0],[1,-1,-1,1,0],[0,0,0,0,1]])
    twkInv = np.matmul(realSenMat[0:4,:], naiveMat.T)
    cond = np.linalg.cond(twkInv)
    #print('Condition number of the tweak-matrix inverse is {}'.format(cond))
    logging.info('Condition number of the tweak-matrix inverse is {}'.format(cond))
    # Make the plots
    #print('Plotting...')
    logging.debug('Plotting...')
    # Start by making the DoFs
    DoFs.append('NULL')
    ff, Eul, fs = OSEM2Eul(paramFile, naiveMat, fftParams=fftParams,  nullStream=True)
    for ii in range(len(Eul)):
        ax[0].semilogy(ff, Eul[ii,:], label=DoFs[ii])
    # Repeat for the measured/calculated input matrix
    #Eul_T, fs = OSEM2Eul(paramFile, invMatNorm, nullStream=False)
    ff, Eul, fs = OSEM2Eul(paramFile, invMat.T, fftParams=fftParams,  nullStream=False)
    # Do the FFT
    for ii in range(len(Eul)):
        ax[1].semilogy(ff, Eul[ii,:], label=DoFs[ii])
    # Formatting
    ax[1].set_xlabel('Frequency [Hz]')
    #ax[1].set_ylabel('Post diag ASD [$\mu \mathrm{m}/ \sqrt{\mathrm{Hz}}$ ]')
    #ax[0].set_ylabel('Pre diag ASD [$\mu \mathrm{m}/ \sqrt{\mathrm{Hz}}$ ]')
    ax[1].set_ylabel('Post diag ASD [ cts/$\sqrt{\mathrm{Hz}}$ ]')
    ax[0].set_ylabel('Pre diag ASD [cts/$ \sqrt{\mathrm{Hz}}$ ]')
    ax[1].set_xlim([0.5,1.02])
    ax[0].legend(loc='best')
    for aa in ax:
        aa.grid(True, which='both')
        aa.set_ylim([1e-2,300])
    props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
    ax[1].text(0.15,0.85, 'Condition number = {}'.format(round(cond,1)), transform=ax[1].transAxes, bbox=props, fontsize=22,fontweight='bold')
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.suptitle('Comparison of diagonality before and after diagonalization for {}'.format(par['optic']))
    fig.savefig(figDir+par['optic']+'_diagComp'+'.pdf', bbox_inches='tight')
    return(invMat, phases, realSenMat, sensMat, cond)

def tabulateFit(fitDictFile):
    '''
    Take output of fitSpectra and print a nice ASCII table
    '''
    fil = h5py.File(fitDictFile,'r')
    DoFs = ['POS','PIT','YAW','SIDE']
    tbl = []
    for ii in DoFs:
        q = fil[ii]['pOpt'][0] / fil[ii]['pOpt'][1]
        tbl.append([ii, '{} +/- {}'.format(fil[ii]['pkFreq'][0], roundTo1(np.sqrt(fil[ii]['pCov'][0,0]))),  '{}'.format(round(q,3)) ])
    #print(tabulate.tabulate(tbl, headers=('DoF','Fitted frequency [Hz]','Q')))
    logging.info(tabulate.tabulate(tbl, headers=('DoF','Fitted frequency [Hz]','Q')))
    fil.close()
    return

def roundTo1(xx):
    '''
    round to some number of significant digits
    '''
    return(round(xx, -int(np.floor(np.log10(abs(xx))))))
