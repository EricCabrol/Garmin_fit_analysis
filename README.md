# Garmin_fit_analysis

Analysis of fit files

fit_files
- https://connect.garmin.com/app/activity/21920233191



# Mistral prompt

Je cherche à exploiter un fichier .fit enregistré sur ma montre Garmin, afin de tracer une information d'allure filtrée, et de la comparer avec la vitesse instantanée qui est très bruitée. Je souhaite utiliser un filtre de Kalman. Peux-tu me guider pour écrire un script Python ? 

https://chat.mistral.ai/chat/7a4909ca-3735-4a40-8551-999ef28090ea


## Erreur 

## Solution (proposée par ChatGPT)
https://chatgpt.com/share/699b5a28-9820-8008-8ae2-4384c725233f 
Désactiver 
Sécurité Windows → Contrôle des applications et du navigateur → Smart App Control


# Claude (online) version

https://claude.ai/chat/37c88937-f8cf-4bdd-a5bb-c2ef05b44e02

Prompt
> My objective is to process my running activity. 
> Write a Python script to filter speed and heading data from a Garmin fit file. Use Kalman filter, use a model for which both speed and heading are constant. Create an html report with two plots : one for the raw and filtered speed, one for the raw and filtered path. The second plot shall use an OpenStreetMap background

Result : garmin_kalman_claude.py