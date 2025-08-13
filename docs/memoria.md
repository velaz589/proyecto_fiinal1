# Memoria del proyecto

Aquí se documenta el desarrollo y resultados del proyecto.

Durante los primeros pasos se ha realizado un esfuerzo de analisis de los datos que nos han llegado el numero de NAN es considerable. antes de avanzar mas lejos se procurara decidir sobre los nan y rellenar en los que se pueda. ademas, una vez realizado esto se seguira haciendo un estudio de las variables, transformando todas aquellas variables no numericas en numericas para que los modelos la pueda procesar. 

## 28/07/2025
Estudio de dataframe y creacion de la columna definitiva de la fecha. resulta que existen distintos dias entre la fecha en la que la empresa recibe la queja del cliente y lo deriva a la empresa de destino. Parece que todas las fechas se corresponden con el 2015.

## del 1 al 13 de agosto
Entre esas fechas se ha realizado la limpieza de datos que quedara como sigue:
- Primero. 
    - Se han borrado las columnas que no aportan informacion o aquellas que tienen un numero muy bajo de campos rellenos. 
- Segundo.
    - Se ha aplicado un labelencoder en las columnas para poder manipularlas, en algunas de ellas se ha realizado una codificacion manual. principalmente para tratar con los Nan ya que el labelencoder no los detecta y los pone como una variable mas. 
- Tercero.
 - se ha añadido una nueva columna con el mes y se han parseado las fechas dejandolas solamente en el dia. 
- Cuarto.
 - Se ha aplicado el modelo de Knnneighbors como imputer, para rellenar la columna de issue. no obstante, no se subira nada de ello. 
- Quinto.
    - Se ha rellenado la columna de state y la de zip cogiendo los numeros zip de internet y se han evaluando. se han borrado el resto que no se han podido procesar, unos 400 numeros. 
- Sexto.
 - notese que no se han quitado los valores atipicos(outliers)
