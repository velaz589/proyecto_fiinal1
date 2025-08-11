"""
evaluation.py
Evaluación del modelo y generación de métricas.
"""

# Aquí va el código de evaluación del modelo
# incluiremos aqui el codigo con los distintos modelos.

import pickle
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import fuentes as ft


def guardar_modelo(nombre:str,modelo):

    with open(f'../models/{nombre}', "wb") as archivo_salida:
        pickle.dump(modelo.best_estimator_, archivo_salida)

def recuperacion(nombre:str):

    with open(f'../models/{nombre}', "rb") as archivo_entrada:
        pipeline_importada = pickle.load(archivo_entrada)
    
        print(pipeline_importada)


def metricas(clf,y_test,predictions=False,X_test):
    print(classification_report(y_test,predictions))
    if predictions:
        preds=predictions
    else:
        preds = clf.predict(X_test)

    preds = clf.predict(X_test)

    print("Score del modelo (accuracy):", round(clf.score(X_test,y_test), 3))
    print("Accuracy score:", round(accuracy_score(preds, y_test), 3))
    print("Recall score:", round(recall_score(preds, y_test), 3))
    print("Precision score:", round(precision_score(preds, y_test), 3))
    print("F1 score:", round(f1_score(preds, y_test), 3))
    print("AUC:", round(roc_auc_score(preds, y_test), 3))

    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds=roc_curve(y_test,y_pred_prob)
    ft.plt.plot(fpr, tpr)
    ft.plt.xlim([0.0, 1.0])
    ft.plt.ylim([0.0, 1.0])
    ft.plt.title('ROC curve for client-complaint classifier')
    ft.plt.xlabel('False Positive Rate (1 - Specificity)')
    ft.plt.ylabel('True Positive Rate (Sensitivity)')
    ft.plt.grid(True)
    