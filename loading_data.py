import numpy as np
import pandas as pd
import os
import json
import glob
import sys
import ktrain

print(ktrain.__version__)


sys.path.insert(0, "../")
root_path = '2020-03-13'

corona_features = {"doc_id": [None], "source": [None], "title": [None],
                  "abstract": [None], "text_body": [None]}

corona_df = pd.DataFrame.from_dict(corona_features)

json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)

#print(json_filenames)

def return_corona_df(json_filenames, df, source):

    for file_name in json_filenames:
        
        row = {'doc_id': None, 'source': None, 'title': None,
                'abstract': None, 'text_body': None}

        with open(file_name) as json_data:
            data = json.load(json_data)

        row['doc_id'] = data['paper_id']
        row['title'] = data['metadata']['title']

        abstract_list = [data['abstract'][x]['text'] for 
                            x in range(len(data['abstract']) - 1)]

        abstract = "\n ".join(abstract_list)

        row['abstract'] = abstract

        text_body_list = [data['body_text'][x]['text'] for
                                x in range(len(data['body_text']) - 1)]

        text_body = '\n '.join(text_body_list)
        row['text_body'] = text_body

        if source == 'b':
            row['source'] = 'BIORXIV'
        elif source == 'c':
            row['source'] = 'COMMON_USE_SUB'
        elif source == 'n':
            row['source'] == 'NON_COMMON_USE'
        elif source == 'p':
            row['source'] == "PMC_CUSTOM_LICENSE"
        
        df = df.append(row, ignore_index=True)
    
    return df

corona_df = return_corona_df(json_filenames, corona_df, 'b')
print(corona_df)
#print(corona_df.isna().sum())
#corona_df.dropna(axis=1)
#print(corona_df.isna().sum())
corona_df = corona_df.drop([0])

texts = corona_df['text_body']
tm = ktrain.text.get_topic_model(texts, n_topics=None, n_features=10000)
tm.print_topics()

tm.build(texts, threshold=0.25)

tm.filter(texts)
tm.filter(corona_df)

print(texts[35])
tm.get_doctopics(doc_ids=[35])
print(tm.topics[np.argmax(tm.get_doctopics(doc_ids=[35]))])
print(tm.print_topics(show_counts=True))

tm.visualize_documents(doc_topics=tm.get_doctopics())

# Let's inspect topics:

print(tm.topics[51])
doc = tm.get_docs(topic_ids=[51], rank=True)[0]
print('DOC_ID: %s' % (doc[1]))
print('TOPIC SCORE: %s' % (doc[2]))
print('TOPIC_ID: %s' % (doc[3]))
print('TEXT: %s' % (doc[0]))

# Let's try to understand what can we get out of transmission/incubation/environmental stability:

transmission_results = tm.search('transmission', case_sensitive=False)
incubation_results = tm.search('incubation', case_sensitive=False)
environmental_results = tm.search('environmental stability', case_sensitive=False)






