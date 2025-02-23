import pandas as pd
import os
project_path =os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

if __name__=="__main__":
    df_base = pd.read_excel(project_path + '/data/ouput_excel/test_.xlsx')
    df_new = pd.read_csv(project_path + '/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_ACL_abstract.csv')
    existing_titles = df_base['title'].tolist()
    df_new = df_new[~df_new['title'].isin(existing_titles)]
    df_new.reset_index(drop=True, inplace=True)
    
    df_new["text"]= "**Title**"+df_new["title"] + "\n\n**Abstract**" + df_new["Abstract"]
    
    df_new["text"] = df_new.apply(lambda x: "**Title**"+x["title"]+"\NNo abstract available" if pd.isna(x["Abstract"]) else x)
    print(df_new["text"].isnull().sum())
    df_new["label"] = ""
    df = df_new[["text", "label"]]
    df_new.to_csv(project_path + '/data/ouput_excel/data_to_annotate.csv', index=False)