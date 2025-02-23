import pandas as pd
import os
project_path =os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


if __name__=="__main__":
    df = pd.read_csv(project_path + '/data/annotated_data/annotation_result_4.csv')
    
    print(df["label"].value_counts())
    
    df_ = df[~df["label"].str.contains("out", case=False, na=False)]
    print(len(df_))
    if not os.path.exists(project_path + '/data/result'):
        os.makedirs(project_path + '/data/result')
    df_.to_csv(project_path + '/data/result/to_be_annotated.csv', index=False)
    