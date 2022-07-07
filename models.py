import numpy as np
import pandas as pd
import faiss
import time
from sentence_transformers import SentenceTransformer
import joblib
from transformers import pipeline

import time

class nlp_models:
    def __init__(self):
        self.question_identifier_pipeline =  pipeline("text-classification" ,r".\Question-identifier")
        self.ans_from_context_pipeline =  pipeline("question-answering" ,r".\Answer-from-context")

    def faiss(self , context_list):
        df = pd.DataFrame(context_list)
        # df = pd.read_csv("./faiss-csv.csv")
        # data=df.headline_text.to_list()
        filename ='distilbert-for-faiss.sav'

        model = joblib.load(filename)
        encoded_data = model.encode(context_list)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        ids = np.array(range(0, len(df)))
        ids = np.asarray(ids.astype('int64'))
        index.add_with_ids(encoded_data, ids)
        faiss.write_index(index, 'abc_news')
        index = faiss.read_index('abc_news')
        print(time.time())
        return index , model 

    def search(self ,query , index , model , context_list):
        t=time.time()
        query_vector = model.encode([query])
        k = 1
        top_k = index.search(query_vector, k)
        #    print('totaltime: {}'.format(time.time()-t))
        return [context_list[_id] for _id in top_k[1].tolist()[0]]

    def answer_from_context_model(self ,ques , context):   
        return self.ans_from_context_pipeline(question= ques, context=context)

    def if_question_model(self , statement):
       return self.question_identifier_pipeline(statement)
