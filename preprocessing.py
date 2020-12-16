#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:18:01 2020

@author: qinggang
"""
import tweepy
import pandas as pd
import time
import os
import json
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np
import reverse_geocoder as rg
from nrclex import NRCLex

os.chdir('/Users/Qinggang1/Documents/graduate/Survmeth727/Final_project')

dates = ['1031', '1102', '1103', '1105', '1106', '1107', '1108', 
         '1109', '1110', '1111', '1112', '1113', '1114', '1115', '1116',
         '1117', '1118', '1119', '1120']
'''
infile = open("1120_covid.pkl", "rb")
data1120 = pickle.load(infile)
infile.close()
'''
coords = pd.read_csv('US_State_Bounding_Boxes.csv')
columns = ['date', 'state', 'compound', 'pos', 'neg', 'neu', 'count']
final_df = pd.DataFrame(columns = columns)
analyzer = SentimentIntensityAnalyzer()
'''
all_states = {}
for i in range(coords.shape[0]):
    place = coords.loc[i, "NAME"]
    all_states[place] = []

i = 0
for twt in data1120:
    if (twt.place.country_code == "US") and (twt.lang == "en"):
        i += 1
        if i % 100 == 0:
            print(i)
        loc_box = twt.place.bounding_box.coordinates[0]
        loc = np.mean(loc_box, axis = 0)
        coordinates = (loc[1], loc[0])
        r = rg.search(coordinates)
        place = r[0]['admin1']
        all_states[place].append(twt)
'''
def sort_data(dates):
    f_name = dates + '_covid.pkl'
    print(dates)
    infile = open(f_name, "rb")
    curr_day = pickle.load(infile)
    infile.close()
    all_states = {}
    for i in range(coords.shape[0]):
        place = coords.loc[i, "NAME"]
        all_states[place] = []
    i = 0
    for twt in curr_day:
        if hasattr(twt, "place") and hasattr(twt.place, "country_code") \
        and hasattr(twt.place, "bounding_box"):
            if (twt.place.country_code == "US") and (twt.lang == "en"):
                i += 1
                if i % 100 == 0:
                    print(i)
                loc_box = twt.place.bounding_box.coordinates[0]
                loc = np.mean(loc_box, axis = 0)
                coordinates = (loc[1], loc[0])
                r = rg.search(coordinates)
                place = r[0]['admin1']
                all_states[place].append(twt)
    outname = 'bystate_' + f_name
    f = open(outname, "wb")
    pickle.dump(all_states, f)
    f.close()
    return all_states


#processing (vader sentiment)

for day in dates:
    all_states = sort_data(day)

    sentiment_state = {}
    for i in all_states.keys():
        sentiment_state[i] = {'pos':[], 'neg':[], 'neu':[], 'compound':[]}
        
    for s in all_states.keys():
        curr_states = all_states[s]
        if len(curr_states) > 0:
            for i in range(len(curr_states)):
                tweet = curr_states[i]
                if hasattr(tweet, "extended_tweet"):
                    text = tweet.extended_tweet["full_text"]
                else:
                    text = tweet.text
                broken_sent = nltk.sent_tokenize(text)
                sent_scores = {'pos':[], 'neg':[], 'neu':[], 'compound':[]}
                for sentence in broken_sent:
                    vs = analyzer.polarity_scores(sentence)
                    for k in vs.keys():
                        sent_scores[k].append(vs[k])
                for k in sent_scores.keys():
                    text_score = sum(sent_scores[k])/len(sent_scores[k])
                    sentiment_state[s][k].append(text_score)
    
    
    for a in sentiment_state.keys():
        count = len(sentiment_state[a]['pos'])
        if count > 0:
            date = day
            pos = sum(sentiment_state[a]['pos'])/len(sentiment_state[a]['pos'])
            neg = sum(sentiment_state[a]['neg'])/len(sentiment_state[a]['neg'])
            neu = sum(sentiment_state[a]['neu'])/len(sentiment_state[a]['neu'])
            comp = sum(sentiment_state[a]['compound'])/len(sentiment_state[a]['compound'])
            final_df = final_df.append({'date': date, 'state': a, 'compound': comp, \
                                        'pos': pos, 'neg': neg, 'neu': neu, 'count': count}, \
            ignore_index = True)

final_df.to_csv(index = False)        



#processing (NRC lexicon)
columns = ['date', 'state', 'fear', 'anger', 'anticip', 'trust', 'surprise', \
           'positive', 'negative', 'sadness', 'disgust', 'joy', 'count']
final_df = pd.DataFrame(columns = columns)

for day in dates:
    all_states = sort_data(day)

    sentiment_state = {}
    for i in all_states.keys():
        sentiment_state[i] = {'fear': [], 'anger': [], 'anticip': [], \
                       'trust': [], 'surprise': [], 'positive': [], \
                       'negative': [], 'sadness': [], 'disgust': [], 'joy': []}
        
    for s in all_states.keys():
        curr_states = all_states[s]
        if len(curr_states) > 0:
            for i in range(len(curr_states)):
                tweet = curr_states[i]
                if hasattr(tweet, "extended_tweet"):
                    text = tweet.extended_tweet["full_text"]
                else:
                    text = tweet.text
                text_object = NRCLex(text)
                text_score = text_object.affect_frequencies
                for k in text_score.keys():
                    if k == 'anticipation':
                        sentiment_state[s]['anticip'].append(text_score[k])
                    else:
                        sentiment_state[s][k].append(text_score[k])
    
    
    for a in sentiment_state.keys():
        count = len(sentiment_state[a]['fear'])
        if count > 0:
            date = day
            fear = sum(sentiment_state[a]['fear'])/len(sentiment_state[a]['fear'])
            anger = sum(sentiment_state[a]['anger'])/len(sentiment_state[a]['anger'])
            anticip = sum(sentiment_state[a]['anticip'])/len(sentiment_state[a]['anticip'])
            trust = sum(sentiment_state[a]['trust'])/len(sentiment_state[a]['trust'])
            surprise = sum(sentiment_state[a]['surprise'])/len(sentiment_state[a]['surprise'])
            positive = sum(sentiment_state[a]['positive'])/len(sentiment_state[a]['positive'])
            negative = sum(sentiment_state[a]['negative'])/len(sentiment_state[a]['negative'])
            sadness = sum(sentiment_state[a]['sadness'])/len(sentiment_state[a]['sadness'])
            disgust = sum(sentiment_state[a]['disgust'])/len(sentiment_state[a]['disgust'])
            joy = sum(sentiment_state[a]['joy'])/len(sentiment_state[a]['joy'])
            final_df = final_df.append({'date': date, 'state': a, 'fear': fear, \
                                        'anger': anger, 'anticip': anticip, \
                                        'trust': trust, 'surprise': surprise, \
                                        'positive': positive, 'negative': negative, \
                                        'sadness': sadness, 'disgust': disgust, \
                                        'joy': joy, 'count': count}, \
            ignore_index = True)

final_df.to_csv('NRC.csv', index = False)   


'''
for twt in data1120:
    if (twt.place.country_code == "US") and (twt.lang == "en"):
        loc_box = twt.place.bounding_box.coordinates[0]
        loc = np.mean(loc_box, axis = 0)
        for ind in range(coords.shape[0]):
            if (loc[0] < coords.loc[ind, "xmax"]) and \
            (loc[0] > coords.loc[ind, "xmin"]) and \
            (loc[1] < coords.loc[ind, "ymax"]) and \
            (loc[1] > coords.loc[ind, "ymin"]): 
                place = coords.loc[ind, "NAME"]
                all_states[place].append(twt)
                break
'''
