import sys, time
from _datetime import datetime
import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from threading import Timer

time_tuple = (2012,  # Year
              9,  # Month
              6,  # Day
              0,  # Hour
              38,  # Minute
              0,  # Second
              0,  # Millisecond
              )


def _win_set_time(time_tuple):
    import win32api
    dayOfWeek = datetime(*time_tuple).isocalendar()[2]
    t = time_tuple[:2] + (dayOfWeek,) + time_tuple[2:]
    win32api.SetSystemTime(*t)


def _linux_set_time(time_tuple):
    import subprocess
    import shlex

    time_string = datetime(*time_tuple).isoformat()

    subprocess.call(shlex.split("timedatectl set-ntp false"))  # May be necessary
    subprocess.call(shlex.split("sudo date -s '%s'" % time_string))
    subprocess.call(shlex.split("sudo hwclock -w"))

#_win_set_time(time_tuple)


def a(val,boll):

     t = np.arange(0.0, 2.0, 0.01)
     s = 1 + np.sin(val*np.pi*t)
     plt.figure(2)
     plt.clf()
     plt.plot(t, s)
     plt.savefig("test.png")
     if boll :
          t = Timer(2,a,[3,False])
          t.start()

a(2,True)
#time.sleep(2)
#a(3,True)
