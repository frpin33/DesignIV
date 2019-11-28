'''
	This file generates Eye diagram and constellation diagram of a C4FM signal
	recorded as IQ data. The IQ data must be in a file named data.iq.
'''

# TODO : unit tests for a few more functions, test parameters at function entry and add error message
# , maybe there is a better solution for documenting the code, fft, power calcul

'''
Desired parameters:
centerFreq = 140.985 * 10 ** 6
bandWidith = 250 * 10 ** 3
level = -70
decimation = 1
interpolation = 10
filterWindowLength = 1000
alpha = 1
symbolFrequency = 4800
nbOfBitsPerAcquisition = 50
nbOfAcquisitions = 20
#                        00   01    10     11
symbolsFreqDeviations = [600, 1800, -600, -1800]
'''

'''   Includes    '''

import numpy as np
import matplotlib.pyplot as plt
import comlib as cl
import time

#Center frequency
centerFreq = 140.985 * 10 ** 6

power = cl.computeRFpower(centerFreq, -40, 30, 12500)
cl.generateGraphs(centerFreq, show = False, save = True, level = power+5, eyeName=eyeName, constName=constName)


#cl.realTimeGraphs(centerFreq)