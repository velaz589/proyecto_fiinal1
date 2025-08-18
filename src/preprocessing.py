"""
preprocessing.py
Procesamiento de datos: escalado, renombrado, descarte de columnas, etc.
"""

# Aquí va el código de procesamiento de datos 
# vamos a poner aqui el codigo que ha sido utilizado para el preprocesamiento de los datos 
# asi copiaremos aqui el codigo resultante de LimpiezaEDA


import pandas as pd
import numpy as np
import regex as re
import skrub
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import colorama
colorama.init()

def main_df_fun():
    '''la funcion toma el DF desde su carpeta 
    devuelve el DF copia en formato df, 
    el DF original 
    el table report 
    tambien recoge una segunda tabla con el codigo postal en principio esta indicado con un archivo. 

    devuelve una tupla(main_df,zip_code y el df copiado)'''
    main_df=pd.read_csv("../data/quejas-clientes.csv")
    zip_code=pd.read_csv("../data/zip_code.csv")
    df=main_df.copy()

    return (main_df,zip_code,df)


def transform(df_aportado=False,id=False,prediccion=False):
    """recibe un DF o no y devuelve el recibido transformado con el label encoder o si no recibe 
    devuelve un DF con las quejas procesadas. 
    devuelve una tupla con el df preparado y una TableReport de skrub. """
    # ahora para la realizacion de la demostracion vamos a reutilizar la mayor parte de este codigo, sin embargo, 
    main_df=main_df_fun()[0]

    df=main_df_fun()[-1]

    if df_aportado is not False and df_aportado is not None:
        df = df_aportado


    encoder = LabelEncoder()

    def transform1(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        # aqui lo que produramos es que si se esta haciendo la prediccion del modelo final ya sabemos que no habra 
        # ni que hacer una nueva columna ni el resto de operaciones ya que ya vendrá con el modelo en el formato deseado. 
        try:
            # esta tranformaicon busca parsear las fechas y luego sacar la diferencia entre el envio de las fechas. 
            #en una columna dejara la diferencia en dias y creara una nueva columna con el mes en el que se recibieron.
            if not prediccion:
                años=pd.to_datetime(main_df["Date received"],yearfirst=True)
                columna_mes=años.dt.month

                df_fechas=pd.to_datetime(main_df["Date sent to company"])-pd.to_datetime(main_df["Date received"])

                df["Date received"]=df_fechas.dt.days
                
                df['mes']=columna_mes

                df.drop('Date sent to company',inplace=True,axis=1)

                return df
            else:
                return df # haremso que retorne el df aunque sinninguna transformacion, podriamos saltarnos este paso.
        except:
            print(Fore.RED +"error en transform1")
    transform1(df,prediccion)

    def transform2(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        # en este caso no necesitamos hacer lo de la prediccion ya que si tiene que darse esta tranformacion.
        try:
            # en esta tranformacion buscamos realizar un cambio a numero de los datos sin usar un label encoder.
            listatemp=[]
            for i in df["Timely response?"].values:
                if i == "Yes":
                    listatemp.append(True)
                else:
                    listatemp.append(False)
            listatemp=pd.Series(listatemp)
            df["Timely response?"]=listatemp.astype(bool)
            return df
        except:
            print("error en tranform2")
    transform2(df,prediccion)

    def transform3(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        # en este caso si que tenemos que introducir la prediccion ya que no habra un unnamed en nuestro df. 
        # tampoco habra la filas esas.
        # y tampoco habra un subissue. no obstante al igual que alguno anterior si la prediccion esta puesta no debera de pasar por aqui.

        try:
            # si no hay una prediccion entonces que se ejecute el codigo sino pasamos.
            if not prediccion:
                df.drop("Unnamed: 0",axis=1,inplace=True)
                df.drop([11730, 13198], inplace=True)
                df.drop("Sub-issue",axis=1,inplace=True)
                return df
            else:
                return df # haremso que retorne el df aunque sinninguna transformacion, podriamos saltarnos este paso.
        except:
            print("error en transform3")
    transform3(df,prediccion)

    def transform4(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        #aqui tambien tendriamos problemas ya que no hace falta realizar los cambios en el data frame.
        # si lo que buscamos es actualizar la prediccion. aunque seria interesante que lo pudieramos sacar por el estado. 
        try:
            columnas_a_cambiar=df[df.columns[4:6]]
            

            zip_code=main_df_fun()[1]
            state=[]
            zip=[]
            for i in zip_code.values:
                resultado = re.findall(r"[A-Z]{2}", i[0])
                state.append(resultado[0])
                x=i[2].replace("to","< z <")
                zip.append(x)

            zip_code=pd.DataFrame({'State': state, 'ZIP Codes': zip})

            listado_estado=[]
            for i in columnas_a_cambiar.values:
                if pd.isna(i[0]) and not pd.isna(i[1]):
                    encontrado=False
                    for x in zip_code.values:
                        z=i[1]
                        y=x[1]
                        if eval(y):
                            listado_estado.append(x[0])
                            encontrado=True
                            break
                    if not encontrado:
                        listado_estado.append(np.nan)
                else:
                    listado_estado.append(i[0])
            

            import random
            listado_zip=[]
            for i in columnas_a_cambiar.values:
                if pd.isna(i[1]) and not pd.isna(i[0]):
                    encontrado=False
                    for x in zip_code.values:

                        
                        if x[0]==i[0]:
                            h=x[1]
                            h=h.replace("z"," ")
                            h=h.split("<")
                            listatemp=[]
                            n1=int(h[0].replace(" ",""))
                            n2=int(h[-1].replace(" ",""))

                            
                            listado_zip.append(random.randint(n1,n2))
                            encontrado=True
                            break
                    if not encontrado:
                        listado_zip.append(np.nan)
                else:
                    listado_zip.append(i[1])

            df["ZIP code"]=listado_zip
            
            
            
            if prediccion:
                pass
            else:
            
                df = df.dropna(subset=['State'],inplace=True)

            return df
        except:
            print("error en transform4")
            
    transform4(df,prediccion)

    def transform5(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        """vamos a realizar el encoding manual de las columnas que tiene nan ya que el label encoder 
        no lo 
        realiza correctamente, devuelve el dataframe con la columna arreglada, recibe un dataframe"""
        # aqui el problema que tenemos es que va a coger los datos del supuesto df, pero claro ahora el df consta de una sola fila 
        # por lo que algunas de las operaciones aqui no seria posibles.
        # vamos a tratar de solucionarlo. 
        
        if not prediccion:
            columna_tratar=df["Sub-product"]
            lista_valores={}

            for i in range(len(columna_tratar.unique())):
                lista_valores[columna_tratar.unique()[i]]=i


            lista_valores[np.nan]=np.nan

            listatemp=[]
            for x in columna_tratar:
                listatemp.append(lista_valores[x])


            df["Sub-product"]=listatemp

            return df

        else: # ahora vamos a ver que sucede si tenemos activada la prediccion
            # en este caso necesitamos que igualmente contar con una lista de valores pero con los valores originales por lo que necesitaremos volver a ejecutar 
            # todos los pasos anteriores como si no hubiera prediccion y luego pasarlo para ver que tendriamos que poner en la linea que se introducira.
            main_df=main_df_fun()[0]
            df_temp=main_df_fun()[-1]
            # cambiamos para que se ejecute todo como si no fuera a ser la prediccion.
            df_temp=main_df_fun()[-1]
            df_temp=transform1(df_temp)
            print("TRanform1\n",df_temp)
            df_temp=transform2(df_temp)
            print("TRanform2\n",df_temp)
            df_temp=transform3(df_temp)
            print("TRanform3\n",df_temp)
            df_temp=transform4(df_temp)
            print("TRanform4\n",df_temp)
            columna_tratar=df_temp["Sub-product"]

            lista_valores={}

            for i in range(len(columna_tratar.unique())):
                lista_valores[columna_tratar.unique()[i]]=i


            lista_valores[np.nan]=np.nan
            # hasta aqui hemos conseguido un dict con la informacion total del dataframe que luego aplicaremos a una unica fila del
            # df 

            listatemp=[]

            columna_tratar=df["Sub-product"] # este df ahora es el que ha venido con las prediccion = True.
            for x in columna_tratar:
                listatemp.append(lista_valores[x])
            # realmente solo tiene una fila podriamos prescindir del bucle for. 

            df["Sub-product"]=listatemp


            

        return df
    
        print("error en transform5")

    transform5(df,prediccion)

    def transform6(df:pd.DataFrame)-> pd.DataFrame:
        # en este caso siempre se hara la transformacion de esta forma.
        try:
            columna_tratar=df["Consumer disputed?"]

            dicionario={np.nan:np.nan,"Yes":1,"No":0}
            listatemp=[]
            for x in columna_tratar:
                listatemp.append(dicionario[x])

            df["Consumer disputed?"]=listatemp
        except:
            print("error en el transform6")

        return df
    transform6(df)

    def transform7(df:pd.DataFrame,prediccion=False)-> pd.DataFrame:
        # en este caso deberemos hacer lo mismo pero nos pasa el mismo problema que en el paso 5 que deberemso de realizar el fit por un lado y el transforma
        # de la columna por otro. 

        try:
            if not prediccion:

                df['Product'] = encoder.fit_transform(df['Product'])
                df['Issue'] = encoder.fit_transform(df['Issue'])
                df['State'] = encoder.fit_transform(df['State'])
                df['Company'] = encoder.fit_transform(df['Company'])
                df['Company response'] = encoder.fit_transform(df['Company response'])
            else:
            # si prediccion esta activo el comportacion debera ser el siguiente.
            # primero formar el df origfinal hasta este punto para hacer el fit. 
                main_df=main_df_fun()[0]
                df_temp=main_df_fun()[-1]
                # cambiamos para que se ejecute todo como si no fuera a ser la prediccion.
                df_temp=main_df_fun()[-1]
                df_temp=transform1(df_temp)
                df_temp=transform2(df_temp)
                df_temp=transform3(df_temp)
                df_temp=transform4(df_temp)
                df_temp=transform5(df_temp)
                df_temp=transform6(df_temp)
                # ahora el df_temp es el df original sin lo de prediccion. 
                df_temp['Product'] = encoder.fit(df_temp['Product'])
                df_temp['Issue'] = encoder.fit(df_temp['Issue'])
                df_temp['State'] = encoder.fit(df_temp['State'])
                df_temp['Company'] = encoder.fit(df_temp['Company'])
                df_temp['Company response'] = encoder.fit(df_temp['Company response'])
                # ahora realizaremos el transform sobre la linea que tenemos de df. 
                df['Product'] = encoder.transform(df['Product'])
                df['Issue'] = encoder.transform(df['Issue'])
                df['State'] = encoder.transform(df['State'])
                df['Company'] = encoder.transform(df['Company'])
                df['Company response'] = encoder.transform(df['Company response'])
                # con esto ya tendriamos aplicado todos los encoders a la linea. 

        except:
            print("error en el transfrom7")

        return df
    transform7(df,prediccion)

    def transform8(df):
        try:
            if not prediccion:
                df_1=df.drop(["Consumer disputed?"],axis=1)
                # creamos unos subconjuntos donde dividimos los datos en train y test
                train = df_1[df_1["Sub-product"].notna()]
                test  = df_1[df_1["Sub-product"].isna()]

                # Entrenar el modelo
                knn_clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
                knn_clf.fit(
                    train[['Complaint ID', 'Product','Issue', 'State', 'ZIP code',
                        'Date received', 'Company', 'Company response', 'Timely response?','mes']],
                    train["Sub-product"]
                )

                # Predecir los valores faltantes
                preds = knn_clf.predict(
                    test[['Complaint ID', 'Product','Issue', 'State', 'ZIP code',
                        'Date received', 'Company', 'Company response', 'Timely response?','mes']]
                )

                # Rellenamos en el DataFrame original
                df.loc[df_1["Sub-product"].isna(), "Sub-product"] = preds
            else:
                pass
        except:
            print("error en el transform8")
        return df
    transform8(df)



    def ultima_predecir(df):
        # esta ultima tranformacion es para rellenar los huecos que no se hubieran rellenado en el marco de meter las cosas en el programa.
        main_df=main_df_fun()[0]
        df_temp=main_df_fun()[-1]
        # cambiamos para que se ejecute todo como si no fuera a ser la prediccion.
        df_temp=main_df_fun()[-1]
        df_temp=transform1(df_temp)
        df_temp=transform2(df_temp)
        df_temp=transform3(df_temp)
        df_temp=transform4(df_temp)
        df_temp=transform5(df_temp)
        df_temp=transform6(df_temp)
        df_temp=transform7(df_temp)
        df_temp=transform8(df_temp)
        # lo que queremos ahora es que el programa que va a hacer la demostracion pueda predecir valores faltantes. 

        
        
        '''knn_clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        knn_clf.fit(
            train[['Complaint ID', 'Product','Issue', 'State', 'ZIP code',
                    'Date received', 'Company', 'Company response', 'Timely response?','mes']],
            train["Sub-product"])'''
        return df_temp
    if prediccion:
        ultima_predecir(df)



    def destransform(df):
        df_decoded=df # esta acabado todavia este modulo.

        df_decoded['Product'] = encoder.inverse_transform(df['Product'])
        df_decoded['Issue'] = encoder.inverse_transform(df['Issue'])
        df_decoded['State'] = encoder.inverse_transform(df['State'])
        df_decoded['Company'] = encoder.inverse_transform(df['Company'])
        df_decoded['Company response'] = encoder.inverse_transform(df['Company response'])
        

    if id:
        df.drop("Complaint ID",axis=1,inplace=True)
    
    

    return df,skrub.TableReport(df)




def train_test(df:pd.DataFrame=False,barajado:bool=False,id:bool=False)-> dict:

    '''coge un dataframe limpiado y devuelve un diccionario. 
    X_train, X_test, y_train, y_test,temp, para_predecir'''
    from sklearn.model_selection import train_test_split
    try:
        if df:
            df=df
            if id:
                df.drop("Complaint ID",axis=1,inplace=True)
        else:
            df=transform()[0]
            if id:
                df.drop("Complaint ID",axis=1,inplace=True)

        temp=df[df["Consumer disputed?"].notna()]
        para_predecir=df[df["Consumer disputed?"].isna()]
        # esto es asi porque en los pipelines que estamos haciendo ya se puede utilizar el shuffle()
        # lo que hace que no sea necesario aqui
        if not barajado:
            temp1=temp.sample(frac=1)
        else:
            pass
        X=temp1.drop("Consumer disputed?",axis=1)
        y=temp1["Consumer disputed?"]

        # haría falta quitar también los valores extraños. 
        

        
        
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.20,
                                                        random_state=55)
                                    
        return {'entrenamiento':(X_train, X_test, y_train, y_test),'temp': temp, 'df_faltantes':para_predecir}
    except:
        print("Ha habido algun error!!")
    






