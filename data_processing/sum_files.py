import pandas as pd
import os
project_path =os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import glob

if __name__=="__main__":
    # df_base = pd.read_excel(project_path + '/data/ouput_excel/test_.xlsx')
    # df_new = pd.read_csv(project_path + '/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_ACL.csv')
    
    # for index, row in df_base.iterrows():
    #     title = row['title']
    #     abstract = row['Abstract']
    #     existing_abstract = df_new.loc[df_new['title'] == title]['Abstract']
    #     if abstract and  existing_abstract.empty:
    #         df_new.loc[df_new['title'] == title, 'Abstract'] = abstract
    # df_new["Publication Date"] = df_new["Publication Date"].apply(lambda x: "" if type(x)==float else int(x))
    # mask  = df_new["Publication Date"].apply(lambda x: True if x=="" else True if x > 2017 else False)
    # df_new_= df_new[mask]
    # df_new_.to_csv(project_path + '/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_ACL_abstract.csv', index=False)
    # existing_titles = df_base['title'].tolist()
    # df_new__ = df_new_[~df_new_['title'].isin(existing_titles)]
    
    # if os.path.exists(project_path + '/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_ACL_abstract.csv'):
    #     df = pd.read_csv(project_path + '/data/ouput_excel/result_WOW_ACM_DBLP_IEEE_ACL_abstract.csv')
    #     df["text"]= df["title"] + "\n\n" + df["Abstract"]
    #     df["label"] = ""
    #     df = df[["text", "label"]]
    #     df.to_csv(project_path + '/data/annotated_data/data_to_annotate.csv', index=False)
    
    files = glob.glob(project_path + '/data/annotated_data/*.csv')
    df = pd.concat([pd.read_csv(file) for file in files])
    df["label"] = df["label"].apply(lambda x: x.split("#") if type(x)!= float else "" )
    set_labels = set()
    for index, row in df.iterrows():
        labels = row['label']
        if type(labels) == list:
            for label in labels:
                set_labels.add(label)
        if type(labels) == str:
            set_labels.add(labels)
    print(set_labels)