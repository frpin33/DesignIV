# Système de surveillance des sites radio

Un projet réalisé pour le cours Design IV -- GEL-3021 pendant la session d'automne 2019 
L'équipe est composé de étudiants de l'université Laval en Génie Électrique et en Génie Information
Les membres de l'équipe : **Edouard Beaulé - Sébastien Buron - Étienne Cyr - Jean-Michel Grenier - Frédérick Pineault**

Afin de limiter des déplacements vers les différents sites isolés permettant une radiocommunication sur l’ensemble du Québec, le Centre de Services Partagés aimerait bonifier les équipements de surveillance du réseau RENIR en équipant certains sites d’un système de surveillance supplémentaire qui vérifiera l’état des canaux de communication et produira une analyse journalière.

## Contenu

Le contenu de ce répertoire se divise en deux sections principales. 

La première section est le code nécessaire pour le fonctionnement de l'ordinateur présent sur le site. Le code est écrit en Python et il est regroupé dans les trois fichiers suivant : **comlib**,**createTask** et **gestionPC**. Chaque fichier contient des commentaires et de la documentation vis-à-vis le rôle de chacune des fonctions écritent. 

La deuxième section est le code nécessaire pour le fonctionnement de l'interface utilisateur. Le code est écrit en JavaScript et en PHP.

## Installation

Le code écrit en langage Python a été testé avec la version 3.7 de Python. Il est reccomandé d'utiliser cette version. Il est possible de télécharger cette version via le [site officiel de Python](https://www.python.org/downloads/). Dans tous les cas, il est déconseiller d'utiliser Python 2 pour faire fonctionner le code.

Une fois l'installation terminé, il est possible d'installer les librairies manquantes avec PIP. Dans une installation standard de Python, PIP est automatiquement installé. La commande PIP a utilisé pour obtenir les librairies nécessaires pour l'éxécution du code est la suivante : 

```
pip install numpy scipy matplotlib seaborn scikit-commpy mysql-connector-python pywin32
```