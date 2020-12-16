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

dates = ['1030', '1031', '1102', '1103', '1105', '1106', '1107', '1108', 
         '1109', '1110', '1111', '1112', '1113', '1114', '1115', '1116',
         '1117', '1118', '1119', '1120']
'''
infile = open("1120_covid.pkl", "rb")
data1120 = pickle.load(infile)
infile.close()
'''
coords = pd.read_csv('US_State_Bounding_Boxes.csv')



     



#processing (NRC lexicon)
columns = ['date', 'state', 'fear', 'anger', 'anticipation', 'trust', 'surprise', \
           'positive', 'negative', 'sadness', 'disgust', 'joy', 'count', \
           'tweet_len', 'num_sentence']
final_df = pd.DataFrame(columns = columns)

for day in dates:
    print(day)
    f_name = 'bystate_{}_covid.pkl'.format(day)
    infile = open(f_name, "rb")
    all_states = pickle.load(infile)
    infile.close()

    sentiment_state = {}
    for i in all_states.keys():
        sentiment_state[i] = {'fear': [], 'anger': [], 'anticipation': [], \
                       'trust': [], 'surprise': [], 'positive': [], \
                       'negative': [], 'sadness': [], 'disgust': [], 'joy': [], \
                       'length': [], 'sentence': []}
        
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
                text_score = text_object.affect_count
                text_length = len(text_object.words)
                sentiment_state[s]['length'].append(text_length)
                sentiment_state[s]['sentence'].append(len(text_object.sentences))
                for k in text_score.keys():
                    sentiment_state[s][k].append(text_score[k])
    
    
    for a in sentiment_state.keys():
        count = len(sentiment_state[a]['fear'])
        if count > 0:
            date = day
            fear = sum(sentiment_state[a]['fear'])/len(sentiment_state[a]['fear'])
            anger = sum(sentiment_state[a]['anger'])/len(sentiment_state[a]['anger'])
            anticip = sum(sentiment_state[a]['anticipation'])/len(sentiment_state[a]['anticipation'])
            trust = sum(sentiment_state[a]['trust'])/len(sentiment_state[a]['trust'])
            surprise = sum(sentiment_state[a]['surprise'])/len(sentiment_state[a]['surprise'])
            positive = sum(sentiment_state[a]['positive'])/len(sentiment_state[a]['positive'])
            negative = sum(sentiment_state[a]['negative'])/len(sentiment_state[a]['negative'])
            sadness = sum(sentiment_state[a]['sadness'])/len(sentiment_state[a]['sadness'])
            disgust = sum(sentiment_state[a]['disgust'])/len(sentiment_state[a]['disgust'])
            joy = sum(sentiment_state[a]['joy'])/len(sentiment_state[a]['joy'])
            length = sum(sentiment_state[a]['length'])/len(sentiment_state[a]['length'])
            sentence = sum(sentiment_state[a]['sentence'])/len(sentiment_state[a]['sentence'])
            final_df = final_df.append({'date': date, 'state': a, 'fear': fear, \
                                        'anger': anger, 'anticipation': anticip, \
                                        'trust': trust, 'surprise': surprise, \
                                        'positive': positive, 'negative': negative, \
                                        'sadness': sadness, 'disgust': disgust, \
                                        'joy': joy, 'count': count, \
                                        'tweet_len': length, 'num_sentence': sentence}, \
            ignore_index = True)

final_df.to_csv('NRC_raw.csv', index = False)   


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
