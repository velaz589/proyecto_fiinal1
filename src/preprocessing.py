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

def main_df():
    main_df=pd.read_csv("../data/quejas-clientes.csv")
    zip_code=pd.read_csv("../data/zip_code.csv")
    df=main_df.copy()

    return (main_df,zip_code,skrub.TableReport(main_df), skrub.TableReport(zip_code),df)


def transform():
    main_df()
    df=main_df()[-1]
    def transform1():
        df_fechas=pd.to_datetime(main_df[0]["Date sent to company"])-pd.to_datetime(main_df[0]["Date received"])

        df["Date received"]=df_fechas

        df.drop('Date sent to company',inplace=True,axis=1)
        return df
    transform1()
    def transform2():
        listatemp=[]
        for i in df["Timely response?"].values:
            if i == "Yes":
                listatemp.append(True)
            else:
                listatemp.append(False)
        listatemp=pd.Series(listatemp)
        df["Timely response?"]=listatemp.astype(bool)
        return df
    transform2()
    def transform3():
        df.drop("Unnamed: 0",axis=1,inplace=True)
        df.drop([11730, 13198], inplace=True)
        df.drop("Sub-issue",axis=1,inplace=True)
        return df
    transform3()
    def transform4():

        columnas_a_cambiar=df[df.columns[4:6]]

        zip_code=main_df()[1]
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
        df = df.dropna(subset=['State'])

        return df,skrub.TableReport(df)
    transform4()



