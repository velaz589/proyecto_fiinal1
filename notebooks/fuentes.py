import pandas as pd
def main_df():
    main_df=pd.read_csv("../data/quejas-clientes.csv")
    return main_df

main_df()