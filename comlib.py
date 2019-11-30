'''
This module is a librairie for C4FM communication analysis
it contains various functions that have been tested by comparing
the results with matlab functions.
'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
from numpy import linalg as la
from sadevice.sa_api import *
import datetime
from scipy.signal import get_window
import seaborn as sns; sns.set() # styling
import commpy
import time
from matplotlib import animation as anim


pi = np.pi


def hamming(N):
	''' 
		Creates a Hamming window for a filter (Matlab like)
		
		Parameters:
		N (int) : Number of points of the window
		
		Returns:
		numpy.array : The window as a numpy array
	'''
	n = np.linspace(0, N-1, N)
	window = 0.54-0.46*np.cos((2*pi*n)/(N-1))
	return window


def lowPassFilter(data, window, fc, fs):
	''' 
		Low pass filters the data with the specified window
		
		Parameters:
		data (numpy.array) : Data to be filtered
		window (numpy.array) : Window to be used for the fitler. The length must be odd.
		fc (int) : Cutoff frequency [Hz]
		fs (int) : Sampling frequency [Hz]
		
		Returns:
		numpy.array : The filtered data in a numpy array
	'''
	wc = 2 * pi * fc / fs                   #Numeric cutoff frequency
	N = len(window)                         #Window length
	M = (N-1)/2
	n = np.linspace(0,N-1,N)
	h = window * wc/pi * np.sinc(wc/pi * (n-M)) #filter windowed impulse response (not causal)
	y = np.convolve(data,h)                 #filtered data
	y = y[N:y.size-N]                       #lifting filter transiant
	return y

def highPassFilter(data, window, fc, fs):
	''' 
		High pass filters the data with the specified window
		
		Parameters:
		data (numpy.array) : Data to be filtered
		window (numpy.array) : Window to be used for the fitler. The length must be odd
		fc (int) : Cutoff frequency [Hz]
		fs (int) : Sampling frequency [Hz]
		
		Returns:
		numpy.array : The filtered data
	'''
	wc = 2 * pi * fc / fs                   #Numeric cutoff frequency
	N = len(window)                         #Window length
	M = int((N-1)/2)                             
	n = np.linspace(0,N-1,N)
	hpb = window * wc/pi * np.sinc(wc/pi * (n-M)) #filter windowed lowpass impulse response (not causal)
	hph = np.concatenate((np.zeros(M),np.array([1]),np.zeros(M))) - hpb #high pass filter impulse response
	y = np.convolve(data,hph)
	y = y[N:y.size-N]                       #lifting filter transiant
	return y

def bandPassFilter(data, window, fcb, fch, fs):
	''' 
		Band pass filters the data with the specified window
		
		Parameters:
		data (numpy.array) : Data to be filtered
		window (numpy.array) : Window to be used for the fitler. The length must be odd
		fcb : Down cutoff frequency [HZ]
		fch : Up cutoff frequency   [HZ]
		fs : Sampling frequency     [Hz]
		
		Returns :
		numpy.array : The filtered data
	'''   
	wcb = 2 * pi * fcb / fs
	wch = 2 * pi * fch / fs
	N = len(window)                         #Window length
	M = int((N-1)/2)                             
	n = np.linspace(0,N-1,N)
	hb = window * wcb/pi * np.sinc(wcb/pi * (n-M)) #filter windowed lowpass impulse response (not causal)
	hb1 = window * wch/pi * np.sinc(wch/pi * (n-M)) #filter windowed lowpass impulse response (not causal)
	hph = np.concatenate((np.zeros(M),np.array([1]),np.zeros(M))) - hb1 #high pass filter impulse response
	h = np.convolve(hb,hph)
	y = np.convolve(data,h)
	y = y[N:y.size-N]                       #lifting filter transiant
	return y

def derivate(data,dt):
	'''
	Computes the derivative of a dataset, uses a numpy gradient function for that.
	
	Parameters:
	data (numpy.array) : Data to be derivated
	dt : Time step between data samples
	
	Returns:
	numpy.array : The derivated data
	'''
	return np.gradient(data,dt)

def convertToPolar(X,Y):
	'''
	Converts the vectors from normal representation to polar representation (norm and phase)
	
	Parameters:
	X (numpy.array): Real value of the vectors
	Y (numpy.array): Imaginairy value of the vectors
	
	Returns:
	numpy.array, numpy.array : Norm, phase of every vectors
	'''
	norm = np.sqrt(X**2+Y**2)
	phase = np.arctan2(Y,X)
	return norm,phase

def filteredInterpolation(data, factor = 10):
	'''
	Interpolates data with a certain interpolation factor. Appy automatically a interpolation
	filter on the upsampled data. The interpolation filter is two hamming filters with wc=pi/factor in series.
	A little data is cropped to ensure that we do not see filters transient.

	Parameters:
	data (numpy.array) : Data to be interpolated
	factor : Interpolation factor. Default is 10

	Returns:
	numpy.array : Interpolated data
	'''
	data = commpy.utilities.upsample(data, factor)
	data = 0.25*factor/10*lowPassFilter(data, np.convolve(hamming(100), hamming(100)), 1/(2*factor), 1)
	data = np.real(data)
	return data

def underSample(data, L, N, fs):
	'''
	Undersamples the data with the specified factor. Applies a decimation filter before
	undersampling.
	
	Parameters:
	data (numpy.array): data to be undersampled
	L : undersampling factor
	N : Lenght of hamming window for decimation filter. If this
	    value is 0, no filter is applied.
	fs : Saampling frequency [Hz]
	
	Returns:
	numpy.array : The undersampled data
	'''
	if (N != 0):
		wc = np.pi / L
		fc = wc*fs / (2*np.pi)
		data = lowPassFilter(data, hamming(N), fc, fs)
		
	nbOfPoints = int(np.floor(data.size/L))
	underSampledData = np.zeros(nbOfPoints)
	for i in range(0,nbOfPoints-1):
		underSampledData[i] = data[i*L]
	
	return underSampledData

def cumulatePhase(phase):
	'''
	Cumulate the phase along a vector. Sometimes the phase of the IQ vector passes from pi to -pi.
	It does not allow us to derivte the signal properly. This function cumulates the phase to avoid
	this problem.
	
	Paramters:
	phase (numpy.array) : phase vector to be cumulated
	
	Returns:
	numpy.array : Vector containing the cumulated phase.
	'''
	
	#Looking for passages from pi to -pi or from -pi to pi
	indexAdd = np.array([])
	indexSubstract = np.array([])
	for i in range(phase.size-2):
		if phase[i+1] > 2 and phase[i+2] < -2:
			indexAdd = np.append(indexAdd,np.array([i+2]))
		if phase[i+1] < -2 and phase[i+2] > 2:
			indexSubstract = np.append(indexSubstract,np.array([i+2]))
	
	indexEvents = np.concatenate((indexAdd, indexSubstract), axis=None)
	indexEvents = np.sort(indexEvents, axis=None)
	cumulativeAddSub = np.zeros(indexEvents.size)
	if np.count_nonzero(indexAdd == indexEvents[0]) > 0:
		cumulativeAddSub[0] = 1
	elif np.count_nonzero(indexSubstract == indexEvents[0]) > 0:
		cumulativeAddSub[0] = -1
	for i in range(1,indexEvents.size):
		if np.count_nonzero(indexAdd == indexEvents[i]) > 0:
			cumulativeAddSub[i] = cumulativeAddSub[i-1] + 1
		elif np.count_nonzero(indexSubstract == indexEvents[i]) > 0:
			cumulativeAddSub[i] = cumulativeAddSub[i-1] - 1
	
	#Adding or substracing factors of 2*pi
	for i in range(0,cumulativeAddSub.size):
		if (i < cumulativeAddSub.size-1):
			phase[int(indexEvents[i]):int(indexEvents[i+1])] = phase[int(indexEvents[i]):int(indexEvents[i+1])] + cumulativeAddSub[i] * 2*np.pi
		else:
			phase[int(indexEvents[i])::] = phase[int(indexEvents[i])::] + cumulativeAddSub[i] * 2*np.pi
	
	return phase

def findDecisionInstant(frequencyDeviation, dt, symbolDt, symbolsFreqDeviations, maxNumOfDecisions = 0):
	'''
	Finds C4FM optimal decision instants.
	
	Paramters:
	frequencyDeviation (numpy.array) : Frequency deviation (comparing to carrier) in function of time
	dt : Time between samples =  1/sampleTime
	symbolDt : Time between sent symbols ex:1/4800 for 4800 symbols per seconds
	symbolsFreqDeviations (list) : List containing the four frequency deviations of symbols ex:[600,1800,-600,-1800]
	
	Returns:
	numpy.array, numpy.array, numpy.array : 1, 2, 3
	1: Vector containing index of decision times
	2 :Vector containing decision times
	3 :Vector containing deviation frequencys at decision times
	
	[finalDecisionIndexVector, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes]
	'''
	t_fd = dt * np.arange(0,frequencyDeviation.size)
	lastDecisionInstant = 0
	decisionIndexs = np.array([])
	decisionNumber = 1
	for i in range(0,frequencyDeviation.size-1):
		if (t_fd[i+1] >= symbolDt*decisionNumber):
			decisionIndexs = np.append(decisionIndexs, i)
			decisionNumber += 1

	nbOfPossibleDecisionMoments = int(symbolDt/dt)
	minNormOfErrorsVector = 100000000000
	firstDecisionIndex = 0
	for j in range(0,nbOfPossibleDecisionMoments):

		decisionIndexs = decisionIndexs + 1 #Sweeping the possible decision indexs
		if (decisionIndexs[decisionIndexs.size-1] > frequencyDeviation.size-1):
			decisionIndexs = np.delete(decisionIndexs, -1)
		
		#Building vector with freqDeviations at decisions times
		freqDeviationsAtDecisionTimes = np.zeros(decisionIndexs.size)
		for i in range(0,decisionIndexs.size):
			freqDeviationIndex = int(decisionIndexs[i])
			freqDeviationsAtDecisionTimes[i] = frequencyDeviation[freqDeviationIndex]

		#Calculating errors vector for frequency deviation at decision time
		freqDeviationsErrorsAtDecisionTimes = np.zeros(decisionIndexs.size)
		for i in range(0,freqDeviationsAtDecisionTimes.size):
			temporairyErrors = np.abs(symbolsFreqDeviations-freqDeviationsAtDecisionTimes[i])
			error = np.min(temporairyErrors)
			freqDeviationsErrorsAtDecisionTimes[i] = error

		#Calculating errors vector norm to see if it is near the optimal decision time
		normOfErrorsVector = la.norm(freqDeviationsErrorsAtDecisionTimes)
		if (normOfErrorsVector < minNormOfErrorsVector):
			minNormOfErrorsVector = normOfErrorsVector
			firstDecisionIndex = j
			finalDecisionIndexVector = decisionIndexs

	#building time vector and frequency deviation vector at decision times
	t_freqDevAtDecisionTime = np.zeros(finalDecisionIndexVector.size)
	finalFreqDeviationsAtDecisionTimes = np.zeros(finalDecisionIndexVector.size)
	for i in range(0,finalDecisionIndexVector.size):
		t_freqDevAtDecisionTime[i] = t_fd[int(finalDecisionIndexVector[i])]
		finalFreqDeviationsAtDecisionTimes[i] = frequencyDeviation[int(finalDecisionIndexVector[i])]

	if (maxNumOfDecisions != 0):
		return [finalDecisionIndexVector[0:maxNumOfDecisions], t_freqDevAtDecisionTime[0:maxNumOfDecisions], finalFreqDeviationsAtDecisionTimes[0:maxNumOfDecisions]]
	else:
		return [finalDecisionIndexVector, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes]

def plotIQData(I,Q,dt,show = False, save = False, filename = 'iqdata'):
	'''
	Show or save the IQ data sent in parameter
	
	Paramters:
	I (numpy.array) : I data of IQ data
	I (numpy.array) : Q data of IQ data
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'iqdata'
	
	Returns:
	None
	'''
	t_iq = dt*np.arange(0,I.size)
	plt.figure(1)
	plt.title('IQ data')
	plt.plot(t_iq,I, label='I')
	plt.plot(t_iq,Q, label='Q')
	plt.xlabel('Temps [s]')
	plt.ylabel('Amplitude')
	plt.grid(True)
	plt.xlim(0,0.02)
	plt.legend()
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None

def plotIQNormAndPhase(norm,phase,dt,show = False, save = False, filename = 'iqNormPhase'):
	'''
	Show or save the IQ data sent in parameter
	
	Paramters:
	norm (numpy.array) : norm of IQ vector along samples
	phase (numpy.array) : phase of IQ vector along samples
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'iqNormPhase'
	
	Returns:
	None
	'''
	t_np = dt*np.arange(0,norm.size)
	plt.figure(2)
	plt.subplot(211)
	plt.title('IQ Norm and cumulative phase')
	plt.xlabel('Temps [s]')
	plt.ylabel('Amplitude')
	plt.plot(t_np, norm)
	plt.grid(True)
	#plt.xlim(0.1,0.11)
	plt.subplot(212)
	plt.xlabel('Temps [s]')
	plt.ylabel('Phase [Rad]')
	plt.plot(t_np, phase)
	plt.grid(True)
	plt.xlim(0,0.02)
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None

def plotDerivativePhase(derivativePhase,dt,show = False, save = False, filename = 'iqPhaseDerivative'):
	'''
	Show or save the IQ data sent in parameter
	
	Paramters:
	phaseDerivative (numpy.array) : phase derivative of IQ vector along samples
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'iqPhaseDerivative'
	
	Returns:
	None
	'''
	t_pd = dt * np.arange(0,derivativePhase.size)
	plt.figure(3)
	plt.title('Dérivée de la phase du vecteur IQ')
	plt.plot(t_pd, derivativePhase)
	plt.grid(True)
	plt.xlim(0,0.02)
	#plt.ylim(-20000,20000)
	plt.xlabel('Temps [s]')
	plt.ylabel('Rad/s')
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None



def plotDeviationFrequency(frequencyDeviation, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes, symbolsFreqDeviations, dt, show = False, save = False, clear = False, filename = 'freqDeviation'):
	'''
	Show or save the deviation frequency plot.
	
	Paramters:
	frequencyDeviation (numpy.array) : C4FM frequency deviation in fct of time
	t_freqDevAtDecisionTime (numpy.array) : Times of decisions
	finalFreqDeviationsAtDecisionTimes (numpy.array) : Frequency deviations at decisions times
	symbolsFreqDeviations (list) : List containing the four frequency deviations of symbols ex:[600,1800,-600,-1800]
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'freqDeviation'
	
	Returns:
	None
	'''
	t_fd = dt * np.arange(0,frequencyDeviation.size)
	plt.figure(4)
	if (clear == True):
		plt.clf()
	plt.title('Déviation de fréquence en fonction du temps')
	plt.plot(t_fd, frequencyDeviation)
	plt.scatter(t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes, s=10, c='red')
	plt.plot([0, t_fd[-1]], [symbolsFreqDeviations[0], symbolsFreqDeviations[0]], '--g')
	plt.plot([0, t_fd[-1]], [symbolsFreqDeviations[1], symbolsFreqDeviations[1]], '--g')
	plt.plot([0, t_fd[-1]], [symbolsFreqDeviations[2], symbolsFreqDeviations[2]], '--g')
	plt.plot([0, t_fd[-1]], [symbolsFreqDeviations[3], symbolsFreqDeviations[3]], '--g')
	plt.grid(True)
	plt.xlim(0,0.02)
	#plt.ylim(-3980,3980)
	plt.xlabel('Temps [s]')
	plt.ylabel('Hz')
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None

def plotEyeDiagram(frequencyDeviation, finalDecisionIndexVector, symbolsFreqDeviations, symbolDt, dt, nbIteration, show = False, save = False, clear = False, filename = 'eyeDiagram', nbOfWindow = 2000):
	'''
	Show or save the deviation frequency plot.
	
	Paramters:
	finalFreqDeviationsAtDecisionTimes (numpy.array) : Frequency deviations at decisions times
	finalDecisionIndexVector (numpy.array) : Index of decision moments in the frequencyDeviation array
	symbolsFreqDeviations (list) : List containing the four frequency deviations of symbols ex:[600,1800,-600,-1800]
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'eyeDiagram'
	
	Returns:
	None
	'''
	t_fd = dt * np.arange(0,frequencyDeviation.size)
	nbOfSamplePerWindow = int(2*symbolDt/dt)
	maxNbOfWindow = np.floor(frequencyDeviation.size/nbOfSamplePerWindow)-5
	if (nbOfWindow > maxNbOfWindow):
		nbOfWindow = int(np.floor(maxNbOfWindow))
	plt.figure(5)
	if (clear == True):
		plt.clf()
	if nbIteration == 0 :
		plt.clf()
	for i in range(0,nbOfWindow):
		plt.plot(t_fd[0:int(finalDecisionIndexVector[i+2]-finalDecisionIndexVector[i])], frequencyDeviation[int(finalDecisionIndexVector[i]):int(finalDecisionIndexVector[i+2])], 'b-', linewidth=0.5)
	plt.plot([t_fd[int(symbolDt/dt)],t_fd[int(symbolDt/dt)]], [-4000,4000], '--r')
	plt.plot([0, 2*symbolDt],[symbolsFreqDeviations[0], symbolsFreqDeviations[0]], '--g')
	plt.plot([0, 2*symbolDt],[symbolsFreqDeviations[1], symbolsFreqDeviations[1]], '--g')
	plt.plot([0, 2*symbolDt],[symbolsFreqDeviations[2], symbolsFreqDeviations[2]], '--g')
	plt.plot([0, 2*symbolDt],[symbolsFreqDeviations[3], symbolsFreqDeviations[3]], '--g')
	plt.grid()
	plt.ylim(-5000,5000)
	plt.minorticks_on()
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.xlabel('Temps(s)')
	plt.ylabel('fréquence [Hz]')
	plt.title("Diagramme de l'oeil")
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None

def plotConstellationDiagram(finalFreqDeviationsAtDecisionTimes, symbolsFreqDeviations, nbIteration, show = False, save = False, clear = False, filename = 'constellationDiagram'):
	'''
	Show or save the deviation frequency plot.
	
	Paramters:
	frequencyDeviation (numpy.array) : C4FM frequency deviation in fct of time
	finalDecisionIndexVector (numpy.array) : Index of decision moments in the frequencyDeviation array
	symbolsFreqDeviations (list) : List containing the four frequency deviations of symbols ex:[600,1800,-600,-1800]
	dt : Time between samples =  1/sampleTime
	show (bool) : Show the figure or not default is False
	save (bool) : Save the figure (with specified filename) default is False
	filename (String) : Filename to save the figure default is 'constellationDiagram'
	
	Returns:
	None
	'''
	plt.figure(6)
	if (clear == True):
		plt.clf()
	if nbIteration == 0 :
		plt.clf()
	plt.title('Diagramme de constellation linéaire')
	plt.xlabel("Déviation de fréquence à l'instant de décision [Hz]")
	plt.grid()
	plt.minorticks_on()
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	yValues = np.zeros(finalFreqDeviationsAtDecisionTimes.size)
	plt.scatter(finalFreqDeviationsAtDecisionTimes, yValues, s=3, c = 'blue')
	plt.plot([symbolsFreqDeviations[0], symbolsFreqDeviations[0]],[-1, 1], '--g')
	plt.plot([symbolsFreqDeviations[1], symbolsFreqDeviations[1]],[-1, 1], '--g')
	plt.plot([symbolsFreqDeviations[2], symbolsFreqDeviations[2]],[-1, 1], '--g')
	plt.plot([symbolsFreqDeviations[3], symbolsFreqDeviations[3]],[-1, 1], '--g')
	plt.yticks([])
	#plt.ylim([-1,1])
	if (save == True):
		plt.savefig(filename + '.png')
	if (show == True):
		plt.show()
	return None


def plotfft(centerFreq, level, nbOfFrames, bandWidith, filename):
	handle = sa_open_device()["handle"]

	sa_config_acquisition(handle, SA_MIN_MAX, SA_LIN_SCALE)
	sa_config_center_span(handle, centerFreq, bandWidith)
	sa_config_level(handle, level)
	sa_config_sweep_coupling(handle, 1000, 1000, True)
	sa_config_real_time(handle, 100, 30)



	sa_initiate(handle, SA_REAL_TIME, 0)
	query = sa_query_sweep_info(handle)
	sweep_width = query["sweep_length"]
	rt2 = np.zeros((sweep_width,nbOfFrames))
	for i in range(nbOfFrames):
		a = (sa_get_real_time_frame(handle))
		rt2[:,i] = a["sweep"]


	sa_abort(handle)
	sa_close_device(handle)

	dbmValues = []
	for i in range(0,sweep_width):
		value = np.mean(rt2[i,:])
		dbmValues.append(value)

	frequences = np.linspace(centerFreq - bandWidith, centerFreq + bandWidith, sweep_width)
	plt.clf()
	plt.plot(frequences/10**6, dbmValues)
	plt.title('Spectre autour de la porteuse')
	plt.xlabel("fréquence [MHz]")
	plt.ylabel("Puissance [dbm]")
	plt.grid()
	plt.minorticks_on()
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.savefig(filename + '.png')

	

def acquire_iq(centerFreq, bandwidth, nbOfSamples, level, decimation, dt):
	'''
	Acquires some IQ data.

	Parameters:
	centerFreq : Center frequency (Hz)
	bandwidith : Bandwidth desired (there is a prefilter made by the radio)
	nbOfSamples : Number of samples desired
	decimation : The sample rate will be 486111.111111111/decimation
	dt : 1/sampleRate
	level : Expected power of the signal [dbm]. It is the equivalent of setting the reference level on spike to maximise
	        the precision.

	Returns:
	np.array of complex numbers : The IQ data. I = np.real(iq), Q = np.real(iq).
	'''

	#Greater number of samples because some data is cropped
	nbOfSamples = int(nbOfSamples + int(nbOfSamples + 0.02/dt))
	span = 250e3 #This is ignored

	handle = sa_open_device()["handle"]
	
	# Configure device
	sa_config_center_span(handle, centerFreq, span)
	sa_config_gain_atten(handle, SA_AUTO_ATTEN, SA_AUTO_GAIN, True)
	sa_config_level(handle, level)
	sa_config_IQ(handle, decimation, bandwidth);
	
	# Initialize
	sa_initiate(handle, SA_IQ, 0);
	return_len = sa_query_stream_info(handle)["return_len"]
	
	# Stream IQ
	iq = np.zeros(nbOfSamples)

	data = sa_get_IQ_data_unpacked(handle, nbOfSamples, True)
	iq = data["iq_data"]
	
	# Close device
	sa_close_device(handle)

	#Croping the 0.1 first seconds because it is not right
	iq = iq[int(0.02/dt)::]
	
	return iq

def readIQFile(fileName):
	''' Reading the IQ file  '''

	with open(fileName, "rb") as f:
			#getting the data from the file...
			f.seek(0, 0)
			data = np.fromfile(f, 'int16')
			data = data[0::2] + 1j * data[1::2]
			I = np.real(data)
			Q = np.imag(data)
	return [I,Q]

def rrcosFilter(data, filterWindowLength, alpha, symbolDt, sampleRate):

	[time_rrc, h_rrc] = commpy.filters.rrcosfilter(filterWindowLength, alpha, symbolDt, sampleRate)
	data = np.convolve(data, h_rrc)
	data = data[filterWindowLength:data.size - filterWindowLength]
	return data

def generateGraphs(centerFreq, symbolsFreqDeviations = [600, 1800, -600, -1800], level = -70,  bandWidith = 12500, \
	symbolFrequency = 4800, nbOfSymbolsPerAcquisition = 50, nbOfAcquisitions = 20, filterWindowLength = 1000, alpha = 1,\
	interpolation = 5, decimation = 1, fftbandWidith = 250e3, eyeDiagram = True, constellationDiagram = True, frequencyDeviationDiagram = False,\
	fft = True, save = True, show = False, eyeName="eyeDiagram" , constName="constellationDiagram"):
	'''
	Generates graphics to analyse the data quality of the c4fm signal. It automaticaly calls the API to acquire the required
	data. The reference level [dbm] is automatically set according to power calcul result. Here are other parameters
	you can play with:

	:param level: Expected power of the signal [dbm]
	:param centerFreq: Center frequency of the signal [Hz]
	:param bandWidith: BandWidith of the signal [Hz]
	:param symbolFrequency: Frequency of the symbols ex:4800 symbols/sec
	:param decimation: Decimation factor sent to the radio. Just use 1 so the sampling frequency is at its maximum.
	:param nbOfSymbolsPerAcquisition: Nb of samples recorded for every acquisitions made by the radio.
	:param nbOfAcquisitions: Nb of different acquisitions made by the radio.
	:param filterWindowLength: Length of the window used for the rrcos filter.
	:param alpha: Alpha parameter for the rrcos filter.
	:param interpolation: Interpolation factor so we have a better resolution.
	:param symbolsFreqDeviations: This is a list containing expected frequency deviations for every 4 symbols.
	:param eyeDiagram: Do we compute the eye diagram or not. Boolean.
	:param constellationDiagram: Do we compute the constellation diagram or not. Boolean.
	:param frequencyDeviation: Do we compute the frequency deviation  diagram or not. Boolean.
	:param fft: Do we compute the fft diagram or not. Boolean.
	:param save: Do we save the desired diagram as png images or not. Boolean
	:param show: Do we show the desired diagrams or not. Boolean.
	:return:
	'''

	#Basic computings on the data
	sampleRate = 486111.111111111 / decimation
	symbolDt = 1 / symbolFrequency
	dt = 1 / sampleRate
	nbOfSamplesPerAcq = nbOfSymbolsPerAcquisition * symbolDt / dt
	interpolatedSampleRate = sampleRate * interpolation
	interpolatedDt = 1 / interpolatedSampleRate

	saveNow = False
	for i in range(nbOfAcquisitions):
		IQ = acquire_iq(centerFreq, bandWidith, nbOfSamplesPerAcq, level, decimation, dt)
		I = np.real(IQ)
		Q = np.imag(IQ)

		# Applying filter on the IQ data
		I = rrcosFilter(I, filterWindowLength, alpha, symbolDt, sampleRate)
		Q = rrcosFilter(Q, filterWindowLength, alpha, symbolDt, sampleRate)

		# Upsampling
		I = filteredInterpolation(I, interpolation)
		Q = filteredInterpolation(Q, interpolation)

		# A little computing on the data
		norm, phase = convertToPolar(I, Q)
		cumulatedPhase = cumulatePhase(phase)
		derivativePhase = derivate(cumulatedPhase, 1) / interpolatedDt
		frequencyDeviation = derivativePhase / (2 * np.pi)
		[finalDecisionIndexVector, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes] \
			= findDecisionInstant(frequencyDeviation, interpolatedDt, symbolDt, symbolsFreqDeviations)

		#Diagrams
		if ((save == True) and (i == nbOfAcquisitions - 1)):
			saveNow = True
		if (eyeDiagram == True):
			plotEyeDiagram(frequencyDeviation, finalDecisionIndexVector, symbolsFreqDeviations, symbolDt, interpolatedDt, i,
				False, saveNow, filename=eyeName)
		if (constellationDiagram == True):
			plotConstellationDiagram(finalFreqDeviationsAtDecisionTimes, symbolsFreqDeviations, i, False, saveNow, filename=constName)
		if (frequencyDeviationDiagram == True):
			plotDeviationFrequency(frequencyDeviation, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes,\
				symbolsFreqDeviations, interpolatedDt, False, saveNow)

	if (show == True):
		plt.show()

	return 1

def realTimeGraphs(centerFreq, symbolsFreqDeviations = [600, 1800, -600, -1800], level = -70,  bandWidith = 12500, \
	symbolFrequency = 4800, nbOfSymbolsPerAcquisition = 50, filterWindowLength = 500, alpha = 1,\
	interpolation = 1, decimation = 1, fftbandWidith = 250e3, eyeDiagram = True, constellationDiagram = True, frequencyDeviationDiagram = False,\
	fft = True):

	# Basic computings on the data
	sampleRate = 486111.111111111 / decimation
	symbolDt = 1 / symbolFrequency
	dt = 1 / sampleRate
	nbOfSamplesPerAcq = nbOfSymbolsPerAcquisition * symbolDt / dt
	interpolatedSampleRate = sampleRate * interpolation
	interpolatedDt = 1 / interpolatedSampleRate

	while (True):
		IQ = acquire_iq(centerFreq, bandWidith, nbOfSamplesPerAcq, level, decimation, dt)
		I = np.real(IQ)
		Q = np.imag(IQ)

		# Applying filter on the IQ data
		I = rrcosFilter(I, filterWindowLength, alpha, symbolDt, sampleRate)
		Q = rrcosFilter(Q, filterWindowLength, alpha, symbolDt, sampleRate)

		# Upsampling
		I = filteredInterpolation(I, interpolation)
		Q = filteredInterpolation(Q, interpolation)

		# A little computing on the data
		norm, phase = convertToPolar(I, Q)
		cumulatedPhase = cumulatePhase(phase)
		derivativePhase = derivate(cumulatedPhase, 1) / interpolatedDt
		frequencyDeviation = derivativePhase / (2 * np.pi)
		[finalDecisionIndexVector, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes] \
			= findDecisionInstant(frequencyDeviation, interpolatedDt, symbolDt, symbolsFreqDeviations)

		# Diagrams
		if (eyeDiagram == True):
			plotEyeDiagram(frequencyDeviation, finalDecisionIndexVector, symbolsFreqDeviations, symbolDt,
						   interpolatedDt,
						   True, False, True)

		if (constellationDiagram == True):
			plotConstellationDiagram(finalFreqDeviationsAtDecisionTimes, symbolsFreqDeviations, True, False, True)
			time.sleep(5)
		if (frequencyDeviationDiagram == True):
			plotDeviationFrequency(frequencyDeviation, t_freqDevAtDecisionTime, finalFreqDeviationsAtDecisionTimes, \
								   symbolsFreqDeviations, interpolatedDt, True, False, True)
			time.sleep(5)

	return 1

def findSymbols(level, centerFrequency, symbolFrequency, decimation, nbOfSymbols):
	#TODO
	return 1

def computeRFpower(centerFreq, level, nbOfFrames, bandWidith):
	handle = sa_open_device()["handle"]

	sa_config_acquisition(handle, SA_MIN_MAX, SA_LIN_SCALE)
	sa_config_center_span(handle, centerFreq, bandWidith)
	sa_config_level(handle, level)
	sa_config_sweep_coupling(handle, 3000, 300, True)
	sa_config_real_time(handle, 100, 30)

	sa_initiate(handle, SA_REAL_TIME, 0)

	rt = []
	for i in range(nbOfFrames):
		a = (sa_get_real_time_frame(handle))
		rt.append(10**(a["sweep"]/10))


	sa_abort(handle)
	sa_close_device(handle)

	rssi_dBm = 10 * np.log10(np.mean(rt))

	return rssi_dBm
