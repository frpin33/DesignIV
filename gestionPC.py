'''
Le fichier gestionPC est un module qui permet l'écriture et la lecture de la base de donnée
Il interagit avec le module comlib pour obtenir l'information demandé par l'utilisateur
Le but principal de ce module est :
    Déterminer, avec la base de donnée, les sites (fréquences) à observer
    Gérer un horaire de lecture/analyse 
    Produire une analyse de qualité (oeil, constellation, fft) et une mesure de puissance pour chaque site listé
    Enregistrer l'analyse sur la base de donnée
    Garantir le succès de l'enregistrement
'''

from threading import Timer
import time, datetime, os, ftplib
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
import comlib as cl

def ecritureHoraire(firstInit, tabHoraire, currentTime):
    '''
    Cette fonction permet de gérer l'ajout de nouvelle plage horaire
    Elle transforme les heures en seconde et les ajoutent dans un tableau
    Elle écrit dans un fichier les plages programmées pour éviter de lancer la même plage plusieurs fois 
    Le fichier est effacé à l'aide du paramètre firstInit
    Param:
        firstInit (bool) : Détermine si le fichier qui contient l'horaire doit être réinitialisé
        tabHoraire (list) : Contient la liste de toutes les plages horaires
        currentTime : L'heure de référence pour le calcul des secondes

    Return:
        tabTempsTimer (list) : Liste de plage horaire en seconde à programmer

    '''

    heureSec = (int(currentTime.strftime("%H"))*3600) 
    minSec = (int(currentTime.strftime("%M"))*60) 
    tpsCurrentSec = heureSec + minSec + int(currentTime.strftime("%S"))

    tabTempsTimer = []
    
    #Le fichier se trouve sur le bureau de l'ordinateur
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    strFile = desktop + "\\Horaire.txt"
    
    #Le contenu du fichier est effacé 
    if firstInit : 
        f = open(strFile, 'w+')
        for obj in tabHoraire:
            strDate = obj[1].strftime("%m-%d-%Y %H:%M:%S") + "\n"
            f.writelines(strDate)

            t_heureSec = (int(obj[1].strftime("%H"))*3600) 
            t_minSec = (int(obj[1].strftime("%M"))*60)
            tmpHoraireSec = t_heureSec + t_minSec +  int(obj[1].strftime("%S"))

            tps = tmpHoraireSec - tpsCurrentSec
            if tps > 0 :  
                tabTempsTimer.append(tps)

    #Les plages horaires déjà présentent dans le fichier ne sont pas effacées
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

                if tps > 0 : 
                    tabTempsTimer.append(tps)
                
    return tabTempsTimer

def getTime():
    '''
    Cette fonction va chercher l'heure et la date sur la base de donnée

    Param :
        None

    Return:
        Time (datetime) : L'heure et la date
    '''
    
    cmdGetTemps = "SELECT CURRENT_TIMESTAMP"

    #On boucle pour assurer la connection
    while(1) :
        try:
            cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
            break
        except:
            time.sleep(5)
    
    cursor = cnx.cursor()

    cursor.execute(cmdGetTemps)
    time = cursor.fetchall()
    cnx.close()

    return time[0][0]

def getTableNTime():
    '''
    Cette fonction récupère la date, l'heure, les sites à lire, les fréquences des sites, 
    le numéro du répeteur et l'horaire de la journée 

    Param :
        None

    Return :
        currentTime (datetime) : L'heure et la date  
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        tableParam (list) : Tableau contenant les sites avec leur fréquence et leur répeteur associer
        tableHoraire (list) : Tableau contenant les plages horaires de la journée

    '''

    currentTime = getTime()
    strCurrentTime = str(currentTime).split(".")[0] 

    cmdGetParam = "SELECT parametres_renir.Numero_RENIR, canaux_renir.Frequence_TX, canaux_renir.Repeteur \
 FROM liste_sites\
 LEFT JOIN parametres_renir ON liste_sites.Numero_RENIR=parametres_renir.Numero_RENIR\
 LEFT JOIN canaux_renir ON parametres_renir.Numero_RENIR=canaux_renir.Numero_RENIR\
 WHERE liste_sites.Numero_systeme=1"

    cmdHoraire = "SELECT * FROM horaire_mesures WHERE Timestamp BETWEEN '" + str(currentTime.date()) + " 00:00:00' AND '" + str(currentTime.date()) +  " 23:59:59' AND Numero_systeme=1"

    tableParam = []
    tableHoraire = []

    #On boucle pour assurer la connection
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
    '''
    Cette fonction appel le module comlib pour obtenir la puissance ainsi que les 3 graphiques (oeil, constellation, fft)

    Param :
        centerFreq : Fréquence central du site à mesurer (Hz)
        site : Numéro du site à mesurer
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        pathPanda (string) : Chemin d'enregistrement, sur l'ordinateur, vers le dossier qui contient les graphiques
        show (bool) : Détermine si on affiche les graphiques
        save (bool) : Détermine si on enregistre les graphiques

    Return : 
        power :  Puissance maximale mesurée (dBm)
    '''

    #Chemin pour enregistrer les photos sur le même fichier à chaque itération (Aucune sauvegarde sur l'ordinateur)
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
    '''
    Cette fonction permet d'écrire les différentes photos des graphiques générés sur la base de donnée

    Param :
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        site : Numéro du site à mesurer
        pathVm (string) : Chemin d'enregistrement, sur la base de donnée, vers le dossier qui contient les graphiques
        pathPanda (string) : Chemin d'enregistrement, sur l'ordinateur, vers le dossier qui contient les graphiques
        writeEye (bool) : Détermine si on écrit le diagramme de l'oeil  
        writeConst (bool) : Détermine si on écrit le diagramme de la constellation 
        writeFFT (bool) : Détermine si on écrit le diagramme de la FFT

    Return : 
        power :  Puissance maximale mesurée (dBm)
    ''' 
    
    #Chemin pour enregistrer les photos sur le même fichier à chaque itération (Aucune sauvegarde sur l'ordinateur)
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
    
    #On boucle pour assurer la connection
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
    '''
    Cette fonction permet de lire la base de donnée pour déterminer si les photos ont été écrites

    Param :
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        site : Numéro du site à mesurer
        pathVm (string) : Chemin d'enregistrement, sur la base de donnée, vers le dossier qui contient les graphiques

    Return :  
        tabWrite (list) : Tableau contenant le statut d'écriture des photos. Le status est représenté par un booléen
    '''

    pathVmEye= pathVm + str(site) + "_eye_" + strCurrentTime + ".png"
    pathVmConst = pathVm + str(site) + "_const_" + strCurrentTime + ".png"
    pathVmFFT = pathVm + str(site) + "_fft_" + strCurrentTime + ".png"
    
    #On boucle pour assurer la connection
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
    '''
    Cette fonction écrit sur la base de donnée des lignes d'informations pour l'affichage dans le UI

    Param :
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        site : Numéro du site à mesurer
        puissance : Puissance maximale mesurée (dBm)
        repeteur : Numéro du répeteur
        writePic (bool) : Détermine si on écrit les informations sur les photos
        writePower (bool) : Détermine si on écrit la puissance

    Return : 
        None

    '''

    nameSite = "Proto A"
    nameEye = str(site) + "_eye_" + strCurrentTime + ".png"
    nameConst = str(site) + "_const_" + strCurrentTime + ".png"
    nameFFT = str(site) + "_fft_" + strCurrentTime + ".png"
    numSys = 1

    cmdWritePicture = "INSERT INTO picture_data\
 (monitoringSite, site, date, filenameEye,filenameConst, filenameOther)\
 VALUES('%s','%i','%s','%s','%s','%s')\
 " %(nameSite,int(site),strCurrentTime,nameEye,nameConst,nameFFT)

    cmdWritePuissance = "INSERT INTO surveillance_sites\
 (Numero_systeme, Numero_RENIR, Repeteur, Timestamp, RSSI_dBm)\
 VALUES('%i','%i','%i','%s','%f')\
 " %(numSys, int(site), int(repeteur), strCurrentTime, float(puissance))

    #On boucle pour assurer la connection
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
    '''
    Cette fonction permet la lecture de la base de donnée pour vérifier le succès des écritures

    Param :
        strCurrentTime (string) : L'heure et la date sans les millisecondes
        site : Numéro du site à mesurer

    Return : 
        tabWrite (list) : Tableau contenant le statut d'écriture des différents données. Le status est représenté par un booléen
    '''

    cmdSuccesPicture = "SELECT * FROM picture_data WHERE date='" + strCurrentTime + "' and site=" + str(site)
    cmdSuccesPuissance = "SELECT * FROM surveillance_sites WHERE Timestamp='" + strCurrentTime + "' and Numero_RENIR=" + str(site)
    picSucces = [] 
    powerSucces = []

    #On boucle pour assurer la connection
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


def deleteHoraire(strCurrentTime):
    '''
    Cette fonction permet de retirer des anciennes plages horaires de la base de donnée

    Param :
        strCurrentTime (string) : L'heure et la date sans les millisecondes

    Return :
        None
    '''
    cmdDelHoraire = "DELETE FROM horaire_mesures WHERE Timestamp <= '" + strCurrentTime.split(" ")[0]  + "'"

    #On boucle pour assurer la connection
    while(1) :
        try:
            cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
            break
        except:
            time.sleep(5)

    cursor = cnx.cursor()

    try :
        cursor.execute(cmdDelHoraire)
        cnx.commit()
    except:
        cnx.rollback()
    cnx.close()


def sendHoraire():
    '''
    Non utilisé dans le code principal
    Cette fonction permet d'écrire des plages horaires sur la base de donnée
    '''

    c = "INSERT INTO horaire_mesures (Numero_systeme, Timestamp) VALUES (1, '2019-11-27 21:12:00')" #, (1, '2019-11-25 22:45:00')"
    cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic')
    cursor = cnx.cursor()

    try :
        cursor.execute(c)
        cnx.commit()
    except:
        cnx.rollback()
    cnx.close()
    

def mainEvent(firstInit):
    '''
    Cette fonction appelle toutes les fonctions nécessaire pour réaliser les différentes tâches
    Elle s'assure que le travail à faire est complété avec succès
    Elle s'occupe de lancer des timers pour couvrir les plages horaires demandées

    Param:
        firstInit (bool) : Détermine si le fichier qui contient l'horaire doit être réinitialisé

    Return:
        None
    '''

    #Chemin sur la base de donnée que le UI utilise pour afficher les photos
    pathVm = "/var/www/html/pictures/"
    
    #Le dossier d'enregistrement des photos est sur le bureau 
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pathPanda = desktop + "\\LattePanda\\" 

    ret = -1 
    while(ret != 0) : 
        currentTime, strCurrentTime, tabParam, tabHoraire = getTableNTime()
        if type(tabParam) == list :
            ret = 0

    deleteHoraire(strCurrentTime)
    
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

def testMesure(wTimer):
    '''
    Non utilisé dans le code principal
    Cette fonction permet de tester la fonction d'analyse de signal
    wTimer permet d'ajouter une analyse 90 secondes après le lancement de la fonction
    '''
    currentTime = datetime.datetime.now()
    strCurrentTime = str(currentTime).split(".")[0]
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pathPanda = desktop + "\\LattePanda\\" 
    site = 107
    centerFreq = 142.185 * 10 ** 6

    if wTimer :
        print("startTimer")
        b = Timer(90, testMesure, [False])
        b.start()

    getSignalData(centerFreq,site,strCurrentTime, pathPanda)
    site = 114
    centerFreq = 140.985 * 10 ** 6
    getSignalData(centerFreq,site,strCurrentTime, pathPanda)


mainEvent(True)
