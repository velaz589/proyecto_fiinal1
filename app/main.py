'''from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}'''




from fastapi import FastAPI, Body
from pydantic import BaseModel
import sys
import pandas as pd
import random
import numpy as np

sys.path.append('c:/Users/nevaz/Desktop/analisis_de_datos/proyecto_fiinal1/notebooks')
sys.path.append('c:/Users/nevaz/Desktop/analisis_de_datos/proyecto_fiinal1/models')
import fuentes as ft
import evaluation
import preprocessing

app = FastAPI(title="API de predicción de reclamaciones")

class ManualInput(BaseModel):
    Complaint_ID: str
    Product: str
    Sub_product: str
    Issue: str
    State: str
    ZIP_code: int
    Date_received: int
    Company: str
    Company_response: str
    Timely_response: str
    mes: int

'''@app.post("/manual")
def modo_manual(input: ManualInput):
    # Construir el DataFrame con los datos recibidos
    dic = input.model_dump()
    df = pd.DataFrame([dic])
    # Preprocesar y predecir (ajusta según tu pipeline real)
    df_temp = preprocessing.transform(df, prediccion=True)
    if isinstance(df_temp, tuple):
        df_temp = df_temp[0]
    # Elimina columnas innecesarias si existen
    for col in ["Consumer disputed?", "Unnamed: 0"]:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    clf = ft.evaluation.recuperacion('RF3(0.79 acu)')
    prediccion = clf.predict(df_temp)
    resultado = "El cliente reclamará posteriormente" if prediccion == 1 else "El cliente no seguirá adelante con la reclamación"
    return {
        "variables_utilizadas": dic,
        "resultado": resultado
    }

@app.get("/prueba")
def modo_prueba():
    dic = {}
    df = preprocessing.main_df_fun()[-1]
    df.drop("Complaint ID", axis=1, inplace=True)
    años = pd.to_datetime(df["Date received"], yearfirst=True)
    columna_mes = años.dt.month
    df_fechas = pd.to_datetime(df["Date sent to company"]) - pd.to_datetime(df["Date received"])
    df["Date received"] = df_fechas.dt.days
    df['mes'] = columna_mes
    df.drop('Date sent to company', inplace=True, axis=1)
    for i in df.columns:
        temp = random.choice(df[i].dropna().unique())
        dic[i] = temp
    df = pd.DataFrame([dic])
    clf = ft.evaluation.recuperacion('RF3(0.79 acu)')
    df_temp = preprocessing.transform(df, prediccion=True)
    if isinstance(df_temp, tuple):
        df_temp = df_temp[0]
    for col in ["Consumer disputed?", "Unnamed: 0"]:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    prediccion = clf.predict(df_temp)
    resultado = "El cliente reclamará posteriormente" if prediccion == 1 else "El cliente no seguirá adelante con la reclamación"
    return {
        "variables_utilizadas": dic,
        "resultado": resultado
    }'''




#uvicorn app_fastapi:app --reload

# ...existing code...
@app.post("/manual")
def modo_manual(input: ManualInput):
    dic = input.model_dump()
    df = pd.DataFrame([dic])
    df_temp = preprocessing.transform(df, prediccion=True)
    if isinstance(df_temp, tuple):
        df_temp = df_temp[0]
    for col in ["Consumer disputed?", "Unnamed: 0"]:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    clf = ft.evaluation.recuperacion('RF3(0.79 acu)')
    prediccion = clf.predict(df_temp)
    # Convertir prediccion a int nativo
    prediccion_py = int(prediccion[0]) if hasattr(prediccion, "__iter__") else int(prediccion)
    resultado = "El cliente reclamará posteriormente" if prediccion_py == 1 else "El cliente no seguirá adelante con la reclamación"
    # Convertir todos los valores del diccionario a tipos nativos
    dic_py = {k: (int(v) if isinstance(v, (np.integer, np.int64)) else v) for k, v in dic.items()}
    return {
        "variables_utilizadas": dic_py,
        "resultado": resultado
    }
# ...existing code...


'''def modo_prueba():
    dic = {}
    df = preprocessing.main_df_fun()[-1]
    df.drop("Complaint ID", axis=1, inplace=True)
    años = pd.to_datetime(df["Date received"], yearfirst=True)
    columna_mes = años.dt.month
    df_fechas = pd.to_datetime(df["Date sent to company"]) - pd.to_datetime(df["Date received"])
    df["Date received"] = df_fechas.dt.days
    df['mes'] = columna_mes
    df.drop('Date sent to company', inplace=True, axis=1)
    for i in df.columns:
        temp = random.choice(df[i].dropna().unique())
        # Convertir a tipo nativo si es numpy
        if isinstance(temp, (np.integer, np.int64)):
            temp = int(temp)
        elif isinstance(temp, (np.floating, np.float64)):
            temp = float(temp)
        dic[i] = temp
    df = pd.DataFrame([dic])
    clf = ft.evaluation.recuperacion('RF3(0.79 acu)')
    df_temp = preprocessing.transform(df, prediccion=True)
    if isinstance(df_temp, tuple):
        df_temp = df_temp[0]
    for col in ["Consumer disputed?", "Unnamed: 0"]:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    prediccion = clf.predict(df_temp)
    prediccion_py = int(prediccion[0]) if hasattr(prediccion, "__iter__") else int(prediccion)
    resultado = "El cliente reclamará posteriormente" if prediccion_py == 1 else "El cliente no seguirá adelante con la reclamación"
    return {
        "variables_utilizadas": dic,
        "resultado": resultado
    }'''

'''@app.get("/prueba")
def modo_prueba(_=None):
    # aqui se cogeran aleatoriamente los nombres y se procurara hacer una prediccion sobre el listado aleatorio 
    import random
    import pandas as pd
    

    #import fuentes as ft

    dic={}
    df=preprocessing.main_df_fun()[-1]
    df.drop("Complaint ID",axis=1,inplace=True)

    años=pd.to_datetime(df["Date received"],yearfirst=True)
    columna_mes=años.dt.month

    df_fechas=pd.to_datetime(df["Date sent to company"])-pd.to_datetime(df["Date received"])

    df["Date received"]=df_fechas.dt.days

    df['mes']=columna_mes

    df.drop('Date sent to company',inplace=True,axis=1)


    for i in df.columns:
        temp=random.choice(df[i].dropna().unique())
        dic[i]=temp

    df=pd.DataFrame([dic])

    clf=ft.evaluation.recuperacion('RF3(0.79 acu)')
    df_temp=preprocessing.transform(df,prediccion=True)[0]
    df_temp.drop("Consumer disputed?", axis=1,inplace=True)
    
    prediccion=clf.predict(df_temp)
    


    resultado1= "El cliente reclamará posteriormente"
    resultado0=" El cliente no seguira adelante con la reclamación"



    if prediccion ==1:
        dic.pop("Consumer disputed?")
        dic.pop("Unnamed: 0")
        # Antes de return
        texto_dic = "\n".join([f"{k}: {v}" for k, v in dic.items()])

        return {
        "variables_utilizadas": dic,
        "resultado": resultado1}
    else:
        dic.pop("Consumer disputed?")
        dic.pop("Unnamed: 0")
        texto_dic = "\n".join([f"{k}: {v}" for k, v in dic.items()])
        return {
        "variables_utilizadas": dic,
        "resultado": resultado0}'''

@app.get("/prueba")
def modo_prueba(_=None):
    dic = {}
    df = preprocessing.main_df_fun()[-1]
    df.drop("Complaint ID", axis=1, inplace=True)

    años = pd.to_datetime(df["Date received"], yearfirst=True)
    columna_mes = años.dt.month
    df_fechas = pd.to_datetime(df["Date sent to company"]) - pd.to_datetime(df["Date received"])
    df["Date received"] = df_fechas.dt.days
    df['mes'] = columna_mes
    df.drop('Date sent to company', inplace=True, axis=1)

    for i in df.columns:
        temp = random.choice(df[i].dropna().unique())
        # Convertir a tipo nativo si es numpy
        if isinstance(temp, (np.integer, np.int64)):
            temp = int(temp)
        elif isinstance(temp, (np.floating, np.float64)):
            temp = float(temp)
        elif isinstance(temp, np.bool_):
            temp = bool(temp)
        dic[i] = temp

    df = pd.DataFrame([dic])
    clf = ft.evaluation.recuperacion('RF3(0.79 acu)')
    df_temp = preprocessing.transform(df, prediccion=True)
    if isinstance(df_temp, tuple):
        df_temp = df_temp[0]
    for col in ["Consumer disputed?", "Unnamed: 0"]:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    prediccion = clf.predict(df_temp)
    # Convertir prediccion a tipo nativo
    if hasattr(prediccion, "__iter__"):
        prediccion_py = int(prediccion[0])
    else:
        prediccion_py = int(prediccion)
    resultado1 = "El cliente reclamará posteriormente"
    resultado0 = "El cliente no seguira adelante con la reclamación"

    dic.pop("Consumer disputed?", None)
    dic.pop("Unnamed: 0", None)

    if prediccion_py == 1:
        return {
            "variables_utilizadas": dic,
            "resultado": resultado1
        }
    else:
        return {
            "variables_utilizadas": dic,
            "resultado": resultado0
        }