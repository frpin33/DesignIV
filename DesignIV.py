#Pseudo code pour la version final

#Le code doit toujours fonctionner et etre en mesure de relancer certaines séqueances dans un cas d'échec 
	#cas échec probable sont les communications entre la radio et vers la BD
	
#Il doit avoir une gestion des fichiers sauvegardés tous les jours et supprimer les fichiers trop vieux
#Vérifier les photos enregistrées existe (avec les noms) réitérer sur la fonction pour les reproduire

#1. Aller chercher via la BD la liste des sites ainsi que leur fréquence centrale
	#Ouverture du "socket" 
	#Envoi de la requête
	#Attendre la réponse 
#2. Pour chaque site :
	#Réaliser un sweep real time // trame IQ // Autre en fonction de la fréquence centrale -> voir avec Étienne
	#Traiter les requêtes via le code de JM -> Diagramme de l'oeil et constellation
	#Enregistrer les images dans un dossier respectif (la date pourrait être considérer ou tout simplement partir d'une date fictive) 
	#// utilisé les dossiers deja existant pour déterminer le nom approprié
	#Nommer les images selon le nom du site récupérer dans la requête initiale
	
#3. Envoi de l'information à la BD
	#Pertinance de vérifié le fonctionnement? -> Plus tard (idéal 2h plus tard environ autre task nécessaire pour le faire)? Le lendemain?
	
	
#autres notes :
	#Bloquer les windows update futur
	#Faire attention au PATH donner dans les scripts --> donner les reals path pour pas avoir de problème