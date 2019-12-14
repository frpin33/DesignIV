'''
Ce module est une librairie pour faire de l'analyse de communication C4FM
Il contient plusieurs fonction testées et comparées avec les fonctions Matlab
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
	Crée une fenêtre de Hamming pour un filtre (simmilaire à Matlab)
	
	Param:
		N (int) : Nombre de points dans la fenêtre
	
	Return:
		numpy.array : La fenêtre dans un numpy array
	'''
	n = np.linspace(0, N-1, N)
	window = 0.54-0.46*np.cos((2*pi*n)/(N-1))
	return window


def lowPassFilter(data, window, fc, fs):
	''' 
	Filtre, avec un passe bas, les données selon la fenêtre spécifiée
	
	Param:
		data (numpy.array) : Données à filtrer
		window (numpy.array) : Fenêtre à utiliser pour le filtre. La taille doit etre impair.
		fc (int) : Fréquence de coupure [Hz]
		fs (int) : Fréquence d'échantillonnage [Hz]
	
	Return:
		numpy.array : Les données filtrées dans un numpy array
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
	Filtre, avec un passe haut, les données selon la fenêtre spécifiée
	
	Param:
		data (numpy.array) : Données à filtrer
		window (numpy.array) : Fenêtre à utiliser pour le filtre. La taille doit etre impair.
		fc (int) : Fréquence de coupure [Hz]
		fs (int) : Fréquence d'échantillonnage [Hz]
	
	Return:
		numpy.array : Les données filtrées dans un numpy array
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
	Filtre, avec un passe bande, les données selon la fenêtre spécifiée
	
	Param:
		data (numpy.array) : Données à filtrer
		window (numpy.array) : Fenêtre à utiliser pour le filtre. La taille doit etre impair.
		fc (int) : Fréquence de coupure [Hz]
		fs (int) : Fréquence d'échantillonnage [Hz]
	
	Return:
		numpy.array : Les données filtrées dans un numpy array
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
	Calcul la derivée des données, utilise la fonction numpy gradient.
	
	Param:
		data (numpy.array) : Données à dériver
		dt : Échelle de temps entre les échantillons de données
	
	Return:
		numpy.array : Les données dérivées
	'''
	return np.gradient(data,dt)

def convertToPolar(X,Y):
	'''
	Convertis le vecteur de coordonnées cartésiennes vers des coordonnées polaires
	
	Param:
		X (numpy.array): Partie réel du vecteur
		Y (numpy.array): Partie imaginaire du vecteur
	
	Returns:
		numpy.array, numpy.array : coordonnée radiale, coordonnée angulaire du vecteur
	'''
	norm = np.sqrt(X**2+Y**2)
	phase = np.arctan2(Y,X)
	return norm,phase

def filteredInterpolation(data, factor = 10):
	'''
	Interpole les données selon un facteur d'interpolation.
	Applique automatiquement un filtre d'interpolation sur les données sur-échantillonnées
	Le filtre est composé de deux filtres de hamming avec wc=pi/factor en série.
	Une petite portion des données est coupée pour enlever les effets transitoires des filtres

	Param:
		data (numpy.array) : Données à interpoler
		factor : Facteur d'interpolation. 10 par défaut

	Return:
		numpy.array : Données interpolées
	'''
	data = commpy.utilities.upsample(data, factor)
	data = 0.25*factor/10*lowPassFilter(data, np.convolve(hamming(100), hamming(100)), 1/(2*factor), 1)
	data = np.real(data)
	return data

def underSample(data, L, N, fs):
	'''
	Sous-échantillonne les données selon un facteur.
	Applique un filtre de décimation avec le sous-échantillonnage
	
	Param:
		data (numpy.array): Donnée à sous-échantillonner
		L : Facteur de sous-échantillonnage
		N : Taille de la fenêtre de Hamming pour le filtre de décimation. If this
			Si la valeur est de 0, aucun filtre est appliqué.
		fs : Fréquence d'échantillonnage [Hz]
	
	Return:
		numpy.array : Les données sous-échantillonées
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
	Cumule la phase au long d'un vecteur.
	La phase passe parfois de pi à -pi dans un vecteur IQ se qui peut nuir à la dérivé
	Cette fonction permet d'éviter ce problème
	
	Param:
		phase (numpy.array) : Vecteur de phase à cumuler

	Return:
		numpy.array : Vecteur avec la phase cumulé
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
	Trouve la décision C4FM optimale instantannée
	
	Param:
		frequencyDeviation (numpy.array) : Fréquence de déviation selon le temps
		dt : Temps entre les échantillons  =  1/sampleTime
		symbolDt : Temps entre les symboles envoyés ex:1/4800 pour 4800 symboles par secondes
		symbolsFreqDeviations (list) : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
	
	Returns:
		numpy.array, numpy.array, numpy.array : 1, 2, 3
		1: Vecteur contenant l'index des temps de décision
		2 :Vecteur contenant les temps de décision
		3 :Vecteur contenant la fréquence de déviation aux temps de décision
		
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
	Affiche et sauvegarde les données IQ des paramètres donnés
	
	Param:
		I (numpy.array) : Donnée I des données IQ
		Q (numpy.array) : Donnée Q des données IQ
		dt : Temps entre les échantillons =  1/sampleTime
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	Affiche et sauvegarde les données IQ des paramètres donnés
	
	Param:
		norm (numpy.array) : Rayon des données IQ
		phase (numpy.array) : Angle des données IQ
		dt : Temps entre les échantillons =  1/sampleTime
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	Affiche et sauvegarde l'angle dérivée des paramètres donnés
	
	Param:
		derivativePhase (numpy.array) : Angle dérivé du vecteur IQ
		dt : Temps entre les échantillons =  1/sampleTime
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	Affiche et sauvegarde la déviation en fréquence des paramètres donnés
	
	Param:
		frequencyDeviation (numpy.array) : Déviation de la fréquence C4FM en fonction du temps
		t_freqDevAtDecisionTime (numpy.array) : Temps de décision
		finalFreqDeviationsAtDecisionTimes (numpy.array) : Fréquence de déviation au temps de décision
		symbolsFreqDeviations (list) : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
		dt : Temps entre les échantillons =  1/sampleTime
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	Affiche et sauvegarde le diagramme de l'oeil des paramètres donnés
	
	Param:
		frequencyDeviation (numpy.array) : Déviation de la fréquence C4FM en fonction du temps
		finalDecisionIndexVector (numpy.array) : Index des temps de décision selon la fréquence de déviation
		symbolsFreqDeviations (list) : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
		symbolDt : Temps entre les symboles envoyés ex:1/4800 pour 4800 symboles par secondes
		dt : Temps entre les échantillons =  1/sampleTime
		nbIteration : Numéro de l'itération en cours
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		clear (bool) : Détermine si on vide la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	Affiche et sauvegarde le diagramme de constellation des paramètres donnés
	
	Param:
		finalFreqDeviationsAtDecisionTimes (numpy.array) : Fréquence de déviation au temps de décision	
		symbolsFreqDeviations (list) : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
		nbIteration : Numéro de l'itération en cours
		show (bool) : Détermine si on affiche la figure
		save (bool) : Détermine si on sauvegarde la figure
		clear (bool) : Détermine si on vide la figure
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
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
	'''
	Calcul et enregistre la fft à l'aide du récepteur
	
	Param:
		centerFreq : Fréquence centrale (Hz)
		level : Puissance du signal attendue (dBm)
		nbOfFrames : Nombre d'échantillons demandés
		bandWidith : Bande passante désirée
		filename (String) : Nom du fichier d'enregistrement de la figure
	
	Return:
		None
	'''
	
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
	Récolte des données IQ avec le récepteur

	Param:
		centerFreq : Fréquence centrale (Hz)
		bandwidith : Bande passante désirée
		nbOfSamples : Nombre d'échantillons demandés
		level : Puissance du signal attendue (dBm)
		decimation : La fréquence d'échantillonnage sera 486111.111111111/decimation
		dt : Temps entre les échantillons =  1/sampleTime
	

	Return:
		np.array des nombres complexes : Les données IQ. I = np.real(iq), Q = np.real(iq).
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
	''' 
	Lecture du fichier IQ 

	Param:
		filename (string) : nom du fichier à lire
	
	Return:
		List :  Les données I et Q du fichier lu
	'''

	with open(fileName, "rb") as f:
			#getting the data from the file...
			f.seek(0, 0)
			data = np.fromfile(f, 'int16')
			data = data[0::2] + 1j * data[1::2]
			I = np.real(data)
			Q = np.imag(data)
	return [I,Q]

def rrcosFilter(data, filterWindowLength, alpha, symbolDt, sampleRate):
	'''
	Fitre, avec un «Root-Raised-cosine», les données en fonction de la fenêtre

	Param:
		data : Donnée à filtrer
		filterWindowLength : Taille de la fenêtre
		alpha : Paramètre alpha du filtre
		symbolDt : Temps entre les symboles envoyés ex:1/4800 pour 4800 symboles par secondes
		sampleRate : Fréquence d'échantillonnage

	Return:
		numpy.array : Les données filtrées
	'''

	[time_rrc, h_rrc] = commpy.filters.rrcosfilter(filterWindowLength, alpha, symbolDt, sampleRate)
	data = np.convolve(data, h_rrc)
	data = data[filterWindowLength:data.size - filterWindowLength]
	return data

def generateGraphs(centerFreq, symbolsFreqDeviations = [600, 1800, -600, -1800], level = -70,  bandWidith = 12500, \
	symbolFrequency = 4800, nbOfSymbolsPerAcquisition = 50, nbOfAcquisitions = 20, filterWindowLength = 1000, alpha = 1,\
	interpolation = 5, decimation = 1, fftbandWidith = 250e3, eyeDiagram = True, constellationDiagram = True, frequencyDeviationDiagram = False,\
	fft = True, save = True, show = False, eyeName="eyeDiagram" , constName="constellationDiagram"):
	'''
	Génère les graphiques pour l'analyse de qualité des données du signal C4FM.
	Cette fonction permet la communication avec le récepteur.
	Le niveau (level) de référence en dBm est automatiquement configuré en fonction de la puissance mesurer


	Param:
		centerFreq : Fréquence centrale (Hz)
		symbolsFreqDeviations : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
		level  : Puissance du signal attendue (dBm)
		bandWidith : Bande passante désirée
		symbolFrequency : Fréquence des symboles
		nbOfSymbolsPerAcquisition = Nombre de symbole enregistré pour chaque itération de mesure
		nbOfAcquisitions  : Nom de mesure réalisée
		filterWindowLength : Taille de la fenêtre du filtre
		alpha : Paramètre alpha pour le filtre RRCosine
		interpolation : Facteur d'interpolation
		decimation : La fréquence d'échantillonnage sera 486111.111111111/decimation
		fftbandWidith : Bande passante de la fft
		eyeDiagram (Bool) : Détermine si on génère le diagramme de l'oeil
		constellationDiagram (Bool) : Détermine si on génère le diagramme de la constellation
		frequencyDeviationDiagram (Bool) : Détermine si on génère le diagramme de déviation de fréquence
		fft (Bool) : Détermine si on génère le diagramme de la fft
		save (Bool) : Détermine si on sauvegarde les différents graphiques générés
		show (Bool) : Détermine si on affiches les différents graphiques générés
		eyeName: Nom du fichier d'enregistrement du diagramme de l'oeil 
		constName: Nom du fichier d'enregistrement du diagramme de la constellation 

	Return:
		None
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

	'''
	Génère les graphiques en fonction d'une analyse temps réel

	Param:
		centerFreq : Fréquence centrale (Hz)
		symbolsFreqDeviations : Liste qui contient les 4 fréquences de déviation des symboles ex:[600,1800,-600,-1800]
		level  : Puissance du signal attendue (dBm)
		bandWidith : Bande passante désirée
		symbolFrequency : Fréquence des symboles
		nbOfSymbolsPerAcquisition = Nombre de symbole enregistré pour chaque itération de mesure
		filterWindowLength : Taille de la fenêtre du filtre
		alpha : Paramètre alpha pour le filtre RRCosine
		interpolation : Facteur d'interpolation
		decimation : La fréquence d'échantillonnage sera 486111.111111111/decimation
		fftbandWidith : Bande passante de la fft
		eyeDiagram (Bool) : Détermine si on génère le diagramme de l'oeil
		constellationDiagram (Bool) : Détermine si on génère le diagramme de la constellation
		frequencyDeviationDiagram (Bool) : Détermine si on génère le diagramme de déviation de fréquence
		fft (Bool) : Détermine si on génère le diagramme de la fft
		save (Bool) : Détermine si on sauvegarde les différents graphiques générés
		show (Bool) : Détermine si on affiches les différents graphiques générés
		eyeName: Nom du fichier d'enregistrement du diagramme de l'oeil 
		constName: Nom du fichier d'enregistrement du diagramme de la constellation 

	Return:
		None

	'''

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


def computeRFpower(centerFreq, level, nbOfFrames, bandWidith):

	'''
	Calcul de la puissance mesurée avec le récepteur

	Param:
		centerFreq : Fréquence centrale (Hz)
		level : Puissance du signal attendue (dBm)
		nbOfFrames : Nombre d'échantillons demandés
		bandWidith : Bande passante désirée
	

	Return:
		rssi_dBm : Puissance maximale mesurée (dBm)
	'''

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
