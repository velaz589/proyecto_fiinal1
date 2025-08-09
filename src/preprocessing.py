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


def transform(df_aportado=False,id=False):
    """recibe un DF o no y devuelve el recibido transformado con el label encoder o si no recibe 
    devuelve un DF con las quejas procesadas. 
    devuelve una tupla con el df preparado y una TableReport de skrub. """

    main_df=main_df_fun()[0]

    df=main_df_fun()[-1]

    if not df_aportado:
        pass
    else:
        df=df_aportado

    encoder = LabelEncoder()

    def transform1(df:pd.DataFrame)-> pd.DataFrame:

        años=pd.to_datetime(main_df["Date received"],yearfirst=True)
        columna_mes=años.dt.month

        df_fechas=pd.to_datetime(main_df["Date sent to company"])-pd.to_datetime(main_df["Date received"])

        df["Date received"]=df_fechas.dt.days
        
        df['mes']=columna_mes

        df.drop('Date sent to company',inplace=True,axis=1)
        return df
    
    transform1(df)

    def transform2(df:pd.DataFrame)-> pd.DataFrame:
        listatemp=[]
        for i in df["Timely response?"].values:
            if i == "Yes":
                listatemp.append(True)
            else:
                listatemp.append(False)
        listatemp=pd.Series(listatemp)
        df["Timely response?"]=listatemp.astype(bool)
        return df
    transform2(df)

    def transform3(df:pd.DataFrame)-> pd.DataFrame:
        df.drop("Unnamed: 0",axis=1,inplace=True)
        df.drop([11730, 13198], inplace=True)
        df.drop("Sub-issue",axis=1,inplace=True)
        return df
    transform3(df)

    def transform4(df:pd.DataFrame)-> pd.DataFrame:

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
        df["State"]=listado_estado

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

        df = df.dropna(subset=['State'],inplace=True)

        return df
    transform4(df)

    def transform5(df:pd.DataFrame)-> pd.DataFrame:
        """vamos a realizar el encoding manual de las columnas que tiene nan ya que el label encoder 
        no lo 
        realiza correctamente, devuelve el dataframe con la columna arreglada, recibe un dataframe"""
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
    transform5(df)

    def transform6(df:pd.DataFrame)-> pd.DataFrame:
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

    def transform7(df:pd.DataFrame)-> pd.DataFrame:
        try:
        
            df['Product'] = encoder.fit_transform(df['Product'])
            df['Issue'] = encoder.fit_transform(df['Issue'])
            df['State'] = encoder.fit_transform(df['State'])
            df['Company'] = encoder.fit_transform(df['Company'])
            df['Company response'] = encoder.fit_transform(df['Company response'])
        except:
            print("error en el transfrom7")

        return df
    transform7(df)
    def transform8(df):
        try:
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
        except:
            print("error en el transform8")
        return df
    transform8(df)
    def destransform(df):
        df_decoded=df

        df_decoded['Product'] = encoder.inverse_transform(df['Product'])
        df_decoded['Issue'] = encoder.inverse_transform(df['Issue'])
        df_decoded['State'] = encoder.inverse_transform(df['State'])
        df_decoded['Company'] = encoder.inverse_transform(df['Company'])
        df_decoded['Company response'] = encoder.inverse_transform(df['Company response'])
        

    if id:
        df.drop("Complaint ID",axis=1,inplace=True)
    
    

    return df,skrub.TableReport(df)

def train_test(df:pd.DataFrame,barajado=False)-> pd.DataFrame:

    '''coge un dataframe limpiado y devuelve 4 DF en una tupla. 
    _train, X_test, y_train, y_test,temp, para_predecir'''
    from sklearn.model_selection import train_test_split
    try:
        df=transform()[0]
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
        
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.20,
                                                        random_state=55)
    except:
        print("Ha habido algun error")
    return X_train, X_test, y_train, y_test, temp, para_predecir





