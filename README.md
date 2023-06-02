# ClasificadorSk
Clasificador de reclamos de gestión de campus universitario utilizando ML con  sklearn en python

La idea esta proyecto es que utilicen el archivo tipo pickle "clasificador_svm.pkl" para extraer el objeto del tipo 
Clasifica y que utilicen el metodo "clasificar" el cual al pasarle una lista de reclamos (lista de string) te devuelve una lista
con las clasificaciones(tres posibles: maestranza, soporte informatico y secretaria tecnica ) de 
dichos reclamos


Aclaracion para la implementacion en windows: 
Si poseen problemas con la importacion de modulos utilizando VScode REESCRIBIR archivo sttings.json en la carpeta
oculta .vscode colocando:
{
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder};${env:PYTHONPATH}",
        "PATH": "${workspaceFolder};${env:PATH}"
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.python"
    },
    "python.formatting.provider": "none"
}
