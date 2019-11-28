from threading import Timer
import time, datetime, os, ftplib
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
import comlib as cl


####DEFINE LES PATHS AU DÉBUT  + USER FRIENDLY

def ecritureHoraire(firstInit, tabHoraire, currentTime):

    heureSec = (int(currentTime.strftime("%H"))*3600) 
    minSec = (int(currentTime.strftime("%M"))*60) 
    tpsCurrentSec = heureSec + minSec + int(currentTime.strftime("%S"))

    tabTempsTimer = []
       
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    strFile = desktop + "\\Horaire.txt"
    
    if firstInit : 
        f = open(strFile, 'w+')
        for obj in tabHoraire:
            strDate = obj[1].strftime("%m-%d-%Y %H:%M:%S") + "\n"
            f.writelines(strDate)

            t_heureSec = (int(obj[1].strftime("%H"))*3600) 
            t_minSec = (int(obj[1].strftime("%M"))*60)
            tmpHoraireSec = t_heureSec + t_minSec +  int(obj[1].strftime("%S"))

            tps = tmpHoraireSec - tpsCurrentSec
            #if tps > 0 ou tps < 24h 
            tabTempsTimer.append(tps)

    else : 
        f = open(strFile, 'r')
        tab = []
        for line in f :
            tab.append(str(line))
        f.close()
        f = open(strFile, 'a')
        for obj in tabHoraire:
            strDate = obj[1].strftime("%m-%d-%Y %H:%M:%S") + "\n"
            if strDate in tab :
                pass
            else :
                f.writelines(strDate)

                t_heureSec = (int(obj[1].strftime("%H"))*3600) 
                t_minSec = (int(obj[1].strftime("%M"))*60)
                tmpHoraireSec = t_heureSec + t_minSec +  int(obj[1].strftime("%S"))
                tps = tmpHoraireSec - tpsCurrentSec

                #if tps > 0 
                tabTempsTimer.append(tps)
                
    return tabTempsTimer


def getTableNTime():

    currentTime = datetime.datetime.now()
    strCurrentTime = str(currentTime).split(".")[0] 

    cmdGetParam = "SELECT parametres_renir.Numero_RENIR, canaux_renir.Frequence_TX, canaux_renir.Repeteur \
 FROM liste_sites\
 LEFT JOIN parametres_renir ON liste_sites.Numero_RENIR=parametres_renir.Numero_RENIR\
 LEFT JOIN canaux_renir ON parametres_renir.Numero_RENIR=canaux_renir.Numero_RENIR\
 WHERE liste_sites.Numero_systeme=1"

    cmdHoraire = "SELECT * FROM horaire_mesures WHERE Timestamp BETWEEN '" + str(currentTime.date()) + " 00:00:00' AND '" + str(currentTime.date()) +  " 23:59:59' AND Numero_systeme=1"

    tableParam = []
    tableHoraire = []

    while(1) :
        try:
            cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
            break
        except:
            time.sleep(5)
    
    cursor = cnx.cursor()
    
    cursor.execute(cmdGetParam)
    tableParam = cursor.fetchall()

    cursor.execute(cmdHoraire)
    tableHoraire = cursor.fetchall()
    
    cnx.close()

    if tableParam :
        return currentTime, strCurrentTime, tableParam, tableHoraire
    else:
        return 0,0,0,0
    

def getSignalData(centerFreq, site, strCurrentTime, pathPanda, show=False, save=True):

    #pathEye = pathPanda + str(site) + "_eye_current"
    #pathConst = pathPanda + str(site) + "_const_current" 
    #pathFFT = pathPanda + str(site) + "_fft_current"

    pathEye = pathPanda + str(site) + "_eye_" + strCurrentTime.replace(":","")
    pathConst = pathPanda + str(site) + "_const_" + strCurrentTime.replace(":","")
    pathFFT = pathPanda + str(site) + "_fft_" + strCurrentTime.replace(":","")

    power = cl.computeRFpower(centerFreq, -40, 30, 12500)
    cl.plotfft(float(centerFreq), -70, 30, 250000, pathFFT)
    cl.generateGraphs(centerFreq, show=show, save=save, level = power+5, eyeName=pathEye, constName=pathConst)
    
    return power
    
def writePictureFTP(strCurrentTime, site, pathVm, pathPanda, writeEye=True, writeConst=True, writeFFT=True):    
    
    #pathPandaEye = pathPanda + str(site) + "_eye_current.png"
    #pathPandaConst = pathPanda + str(site) + "_const_current.png" 
    #pathFFT = pathPanda + str(site) + "_fft_current.png"
    
    pathPandaEye = pathPanda + str(site) + "_eye_" + strCurrentTime.replace(":","") + ".png"
    pathPandaConst = pathPanda + str(site) + "_const_" + strCurrentTime.replace(":","") + ".png"
    pathPandaFFT = pathPanda + str(site) + "_fft_" + strCurrentTime.replace(":","") + ".png"
    
    pathVmEye = pathVm  + str(site) + "_eye_" + strCurrentTime + ".png"
    pathVmConst = pathVm + str(site) + "_const_" + strCurrentTime + ".png"
    pathVmFFT = pathVm + str(site) + "_fft_" + strCurrentTime + ".png"
    
    cmdSTOREye = "STOR " + pathVmEye
    cmdModEye = "SITE CHMOD 755 " + pathVmEye
    cmdSTORConst = "STOR " + pathVmConst
    cmdModConst = "SITE CHMOD 755 " + pathVmConst
    cmdSTORFFT = "STOR " + pathVmFFT
    cmdModFFT =  "SITE CHMOD 755 " + pathVmFFT
    
    #try catch pour la connection ici
   
    while(1) :
        try:
            session = ftplib.FTP('51.79.55.231','ulaval','Equipe9!')
            break
        except:
            time.sleep(5)
    
    
    if writeEye :
        file = open(pathPandaEye,'rb') 
        session.storbinary(cmdSTOREye, file) 
        file.close()
        session.sendcmd(cmdModEye)

    if writeConst : 
        file = open(pathPandaConst,'rb') 
        session.storbinary(cmdSTORConst, file)
        file.close()    
        session.sendcmd(cmdModConst)

    if writeFFT : 
        file = open(pathPandaFFT,'rb') 
        session.storbinary(cmdSTORFFT, file)
        file.close()    
        session.sendcmd(cmdModFFT)

    session.quit()


def readPictureFTP(strCurrentTime, site, pathVm) :

    pathVmEye= pathVm + str(site) + "_eye_" + strCurrentTime + ".png"
    pathVmConst = pathVm + str(site) + "_const_" + strCurrentTime + ".png"
    pathVmFFT = pathVm + str(site) + "_fft_" + strCurrentTime + ".png"
    
    while(1) :
        try:
            session = ftplib.FTP('51.79.55.231','ulaval','Equipe9!')
            break
        except:
            time.sleep(5)
    
    currentDir = session.nlst(pathVm)    
    session.quit()
    
    tabWrite = [False, False, False]
    if pathVmEye not in currentDir:
        tabWrite[0] = True
    if pathVmConst not in currentDir :
        tabWrite[1] = True
    if pathVmFFT not in currentDir: 
        tabWrite[2] = True
    
    return tabWrite


def writeDataBase(strCurrentTime, site, puissance, repeteur, writePic=True, writePower=True):

    nameSite = "Proto A"
    nameEye = str(site) + "_eye_" + strCurrentTime + ".png"
    nameConst = str(site) + "_const_" + strCurrentTime + ".png"
    nameEmpty = "empty.png"
    numSys = 1

    cmdWritePicture = "INSERT INTO picture_data\
 (monitoringSite, site, date, filenameEye,filenameConst, filenameOther)\
 VALUES('%s','%i','%s','%s','%s','%s')\
 " %(nameSite,int(site),strCurrentTime,nameEye,nameConst,nameEmpty)

    cmdWritePuissance = "INSERT INTO surveillance_sites\
 (Numero_systeme, Numero_RENIR, Repeteur, Timestamp, RSSI_dBm)\
 VALUES('%i','%i','%i','%s','%f')\
 " %(numSys, int(site), int(repeteur), strCurrentTime, float(puissance))

    while(1) :
        try:
            cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
            break
        except:
            time.sleep(5)
    
    
    cursor = cnx.cursor()
    
    if writePic :
        try :
            cursor.execute(cmdWritePicture)
            cnx.commit()
        except:
            cnx.rollback()

    if writePower :
        try :
            cursor.execute(cmdWritePuissance)
            cnx.commit()
        except:
            cnx.rollback()
    
    cnx.close()

def readDataBase(strCurrentTime, site):

    cmdSuccesPicture = "SELECT * FROM picture_data WHERE date='" + strCurrentTime + "' and site=" + str(site)
    cmdSuccesPuissance = "SELECT * FROM surveillance_sites WHERE Timestamp='" + strCurrentTime + "' and Numero_RENIR=" + str(site)
    picSucces = [] 
    powerSucces = []

    while(1) :
        try:
            cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
            break
        except:
            time.sleep(5)
    
    cursor = cnx.cursor()

    cursor.execute(cmdSuccesPicture)
    picSucces = cursor.fetchall()

    cursor.execute(cmdSuccesPuissance)
    powerSucces = cursor.fetchall()
    cnx.close()

    tabWrite = [False, False]


    if not picSucces :
        tabWrite[0] = True

    if not powerSucces :
        tabWrite[1] = True

    return tabWrite


#####Autre
#Gestion de l'heure sur le PC 
#Commentaire et documentation

def sendHoraire():
    c = "INSERT INTO horaire_mesures (Numero_systeme, Timestamp) VALUES (1, '2019-11-27 21:12:00')" #, (1, '2019-11-25 22:45:00')"
    cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
    cursor = cnx.cursor()

    try :
        cursor.execute(c)
        cnx.commit()
    except:
        cnx.rollback()
    cnx.close()
    

#Code pour les return value a ajouter
def mainEvent(firstInit):

    pathVm = "/var/www/html/pictures/"
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pathPanda = desktop + "\\LattePanda\\" 

    ret = -1 
    while(ret != 0) : 
        currentTime, strCurrentTime, tabParam, tabHoraire = getTableNTime()
        if type(tabParam) == list :
            ret = 0

    timers = []
    if tabHoraire :
        timers = ecritureHoraire(firstInit, tabHoraire, currentTime)
        for obj in timers:
            t = Timer(int(obj), mainEvent, [False])
            t.start()
            print("Timer Start")

    for obj in tabParam :
        print("Analysing a frequency")
        site = int(obj[0])
        centerFreq = obj[1] * 10**6
        print(centerFreq)
        repeteur = int(obj[2])
        puissance = getSignalData(centerFreq, site, strCurrentTime, pathPanda)

        ret = [True, True, True]
        while(True in ret) : 
            writePictureFTP(strCurrentTime, site, pathVm, pathPanda, ret[0], ret[1], ret[2])    
            ret = readPictureFTP(strCurrentTime, site, pathVm)

        ret = [True, True]
        while(True in ret) : 
            writeDataBase(strCurrentTime, site, puissance, repeteur, ret[0], ret[1])
            ret = readDataBase(strCurrentTime, site)

    print("My work is done")

def testyTest(a):
    currentTime = datetime.datetime.now()
    strCurrentTime = str(currentTime).split(".")[0]
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pathPanda = desktop + "\\LattePanda\\" 
    site = 107
    centerFreq = 140.985 * 10 ** 6

    if a :
        print("startTimer")
        b = Timer(90, testyTest, [False])
        b.start()

    getSignalData(centerFreq,site,strCurrentTime, pathPanda)






#testyTest(False)
#sendHoraire() 
mainEvent(False)







