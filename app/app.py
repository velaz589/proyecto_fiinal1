"""
app.py
Aplicación web para el modelo entrenado final.
"""

# Aquí va el código de la aplicación web
import gradio as gr
import sys
sys.path.append('../notebooks')

import fuentes as ft 


import gradio as gr
import preprocessing

def modo_manual():  # tus argumentos aquí
    # lógica para modo manual
    #RF3(0.79 acu)
    return "Resultado manual", 0

def modo_prueba(_=None):
    # aqui se cogeran aleatoriamente los nombres y se procurara hacer una prediccion sobre el listado aleatorio 
    import random
    import pandas as pd
    import sys
    sys.path.append('../notebooks')

    import fuentes as ft

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

        return f"Variables utilizadas:\n{texto_dic}\n\n",f"Resultado prueba:\n{resultado1}"
    else:
        dic.pop("Consumer disputed?")
        dic.pop("Unnamed: 0")
        texto_dic = "\n".join([f"{k}: {v}" for k, v in dic.items()])
        return f"Variables utilizadas:\n{texto_dic}\n\n",f"Resultado prueba:\n{resultado0}"
    
    


manual_inputs = [
    gr.Textbox(label="Complaint ID"),
    gr.Textbox(label="Product"),
    gr.Textbox(label="Sub-product"),
    gr.Textbox(label="Issue"),
    gr.Textbox(label="State"),
    gr.Number(label="ZIP code"),
    gr.Number(label="Date received"),
    gr.Textbox(label="Company"),
    gr.Textbox(label="Company response"),
    gr.Textbox(label="Timely response?"),
    gr.Number(label="mes")
]

manual_interface = gr.Interface(
    fn=modo_manual,
    inputs=manual_inputs,
    outputs=["text", "number"],
    title="Manual"
)

prueba_interface = gr.Interface(
    fn=modo_prueba,
    inputs=[gr.Markdown("## Este es el modo prueba. Aquí puedes ver una predicción automática con datos de ejemplo.")],
    outputs=["text","text"],
    title="Prueba"
)



demo = gr.TabbedInterface([manual_interface, prueba_interface], ["Manual", "Prueba"])
demo.launch()