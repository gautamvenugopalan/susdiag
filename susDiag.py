import argparse
import SUSdiagUtils as utils
from SUSdiagUtils import *

parser = argparse.ArgumentParser(description=
        '''Usage:
            python susDiag.py --paramFile <path_to_parameter_file> --initFile <path_to_init_guess_file>
           will use the parameter file and file with the initial guesses to do the following operations:
           1. Download the data for the times, and from the NDS server specified in the param file, downsample it as specified,and save to a .hdf5 file in the Data directory
           2. Make high-res (fft length specified in the param file)spectra and save the figure
           3. Make a time domain plot of the signals, and save it,mainly for diagnostic purposes
           4. Look for peaks in spectra, in the specified frequency ranges
           5. Fit these peaks with Lorentzians, mainly to extract the location of the center
           6. Compute the (complex) TF from all coils to UL.
           7. Build a sensing matrix using the TFs from #6, and the peak positions from #5. Some diagnostics like the ratio of imag/real parts of the matrix elements are checked.
           8. Invert the matrix from #7 to get the matrix that should be used in our suspension controls systems.
        ''')
parser.add_argument('--paramFile', type=str, help='Path to the parameter file specifying which optic to analyze', nargs=1, required=True)
parser.add_argument('--initFile', type=str, help='Path to the file containing the initial guesses', nargs=1, required=True)
args = parser.parse_args()

# Global setup
paramFile = args.paramFile[0]
initFile = args.initFile[0]
par = importParams(paramFile)
if par['optic'] not in os.listdir(globDataDir):
    logging.debug('Making subdirectory for {} in {}'.format(par['optic'], globDataDir))
    os.mkdir(globDataDir+par['optic'])
if par['optic'] not in os.listdir(globFigDir):
    logging.debug('Making subdirectory for {} in {}'.format(par['optic'], globFigDir))
    os.mkdir(globFigDir+par['optic'])
# Define the directory Macros
utils.dataDir = globDataDir + par['optic'] + '/'
utils.figDir = globFigDir + par['optic'] + '/'
# Build the FFT dictionary
fftParams = {}
fftParams['window'] = ('tukey',0.25)
fftParams['tFFT'] = 1024


# Download the data
dlData(paramFile)
# Make the high res spectra after downsampling
highResSpectra(paramFile)
# Make the time and frequency domain plots
fig, ax = plt.subplots(1,1,figsize=(16,9))
fig2, ax2 = plt.subplots(2,3,sharex=True, sharey=True, figsize=(20,12))
plotSpectra(paramFile, fig, ax)
plotTimeSeries(paramFile, fig2, ax2)

# Fit the peaks
fig3, ax3 = plt.subplots(1,1,figsize=(16,9))
fitDict = fitSpectra(paramFile, initFile, fftParams, fig3, ax3, nullStream=True)

# Compute the complex TF from all coils to UL
fig4, ax4 = plt.subplots(3,1,figsize=(16,24), sharex=True)
ff_TF, TF = cplxTF(paramFile, fftParams, fig4, ax4)

# Compute the sensing matrix
fig5, ax5 = plt.subplots(2,1,sharex=True,figsize=(16,16))
invMat, phases, realSenMat, sensMat, cond = calcSensMat(paramFile, fftParams, fitDict, TF, ff_TF, fig5, ax5)

# Make a data file with the results of this analysis for possible later use
try:
    fil = h5py.File(utils.dataDir + 'diagData.hdf5','w')
    logging.debug('Saving the analysis data to {}...'.format(utils.dataDir + 'diagData.hdf5'))
    for k, v in fitDict.items():
        gg = fil.create_group(k)
        for ll, mm in v.items():
            gg.create_dataset(ll, data=mm)
    gg = fil.create_group('Matrices')
    gg.create_dataset('ConditionNumber',data=cond)
    gg.create_dataset('InverseMatrix',data=invMat)
    gg.create_dataset('ComplexMatrix',data=sensMat)
    gg.create_dataset('Phases',data=phases)
    fil.close()
    # Print the fit summary
    tabulateFit(utils.dataDir + 'diagData.hdf5')
except Exception as e:
    logging.warning(e)

# Finally, print the matrix that should go into the EPICS screen
logging.info('Computed matrix that will best diagonalize the sensor signals is')
sympy.pprint(round_expr(sympy.Matrix(invMat.T),3), full_prec=False)
