# Memoria del proyecto

Aquí se documenta el desarrollo y resultados del proyecto.

Durante los primeros pasos se ha realizado un esfuerzo de analisis de los datos que nos han llegado el numero de NAN es considerable. antes de avanzar mas lejos se procurara decidir sobre los nan y rellenar en los que se pueda. ademas, una vez realizado esto se seguira haciendo un estudio de las variables, transformando todas aquellas variables no numericas en numericas para que los modelos la pueda procesar. 

## 28/07/2025
Estudio de dataframe y creacion de la columna definitiva de la fecha. resulta que existen distintos dias entre la fecha en la que la empresa recibe la queja del cliente y lo deriva a la empresa de destino. Parece que todas las fechas se corresponden con el 2015.
Para el entrenamiento de los modelos se ha observado el acuraci, la precisión cno la que el modelo acierta la variable en el modo test. No es tan determinante acertar los que no van a generar problemas como los que si, por lo que se puede aceptar un margen de error mediano o grande, que tampoco se da en los modelos realizados. Por lo que es bastante aceptable. 
el entrenamiento se ha realizado con el complaint id aunque luego se revelarian una molestia. 



## Del 1 al 13 de agosto
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

## Del 13 de agosto al 20
Durante este tiempo se ha acabado de realizarla parte de preprocesado y se ha iniciado el desarrollo de la app. En este momento tengo la dificultad de hacer el diseño de la app y que funcione con el modelo que he estado haciendo. me he dado cuenta que realmente hay muchas variables adicionales que estan dentro de la app y que dificultan el uso de la misma. me estoy planteando hacer otro modelo con menos variables. ya que realmente a la hora querer hacer la comprobacion si necesitamos meter la empresa 
O realizar un diseño de la app que permita ver las variables a seleccionar.


## semana del 25 
ahora me encuentro con el problema de que la demostracion si se deja un monton de opciones no es demasiado ergonomica lo que hace es que sea dificil de utilizar, no es realista su uso si tiene uno que poner el nombre entero de la empresa u otros problemas que dificultarian el uso de la mismo o laposible demostracion. 