'''
Le fichier createTask est lancé à chaque ouverture de l'ordinateur
Son rôle est de créer une tâche qui se lancera tous les jours à Minuit (Heure réel)
La tâche crée lance le fichier gestionPC qui se trouve sur le bureau dans le fichier Design IV
Le LattePanda ne possède pas de batterie CMOS, l'heure lors de son ouverture est donc celle lors de la dernière fermeture
Le LattePanda utilise Internet pour mettre la date et l'heure à jour, mais le tout est impossible lorsqu'il sera sur un site
La tâche crée se fit donc au temps réel (sur la base de donnée) pour partir la tâche à minuit
Le temps sur l'ordinateur est seulement considérer pour ajouter le temps nécessaire avant minuit
'''

import sys, time, os
import datetime
import win32com.client
import mysql.connector


#Requête pour obtenir l'heure de la base de donnée 
#Très similaire à getTime() mais on ne récupère pas la date
cmdGetHeure = "SELECT CURRENT_TIME"

#On boucle pour assurer la connection
while(1) :
     try:
          cnx = mysql.connector.connect(user='fred', password='Equipe9!',host='51.79.55.231',database='clic', connection_timeout=40)
          break
     except:
          time.sleep(10)

cursor = cnx.cursor()
cursor.execute(cmdGetHeure)
time = cursor.fetchall()
cnx.close()

#86400 secondes dans une seule journée, on rajoute 10 secondes par sécurité 
timeB4MIDNIGHT =  86410 - time[0][0].total_seconds() 

scheduler = win32com.client.Dispatch('Schedule.Service')
scheduler.Connect()
root_folder = scheduler.GetFolder('\\')
task_def = scheduler.NewTask(0)

# Create trigger
start_time = datetime.datetime.now() + datetime.timedelta(seconds=timeB4MIDNIGHT)
TASK_TRIGGER_TIME = 2
trigger = task_def.Triggers.Create(TASK_TRIGGER_TIME)
trigger.StartBoundary = start_time.isoformat()

# Create action
TASK_ACTION_EXEC = 0
action = task_def.Actions.Create(TASK_ACTION_EXEC)
action.ID = 'Start the main python program'

#Chemin vers le dossier qui contient le fichier gestionPC
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
path = desktop + "\\DesignIV\\gestionPC.py"

action.Path = path

# Set parameters
task_def.RegistrationInfo.Description = 'Cette tâche permet l\'exécution journalière du code python gestionPC'
task_def.Settings.Enabled = True
task_def.Settings.StopIfGoingOnBatteries = False

# Register task
# If task already exists, it will be updated
TASK_CREATE_OR_UPDATE = 6
TASK_LOGON_NONE = 0
root_folder.RegisterTaskDefinition(
    'Task DesignIV',  # Task name
    task_def,
    TASK_CREATE_OR_UPDATE,
    '',  # No user
    '',  # No password
    TASK_LOGON_NONE)
