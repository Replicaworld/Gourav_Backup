#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()


# In[2]:


from pymongo import MongoClient
from pymongo import UpdateOne

def getMongoClient():
    hosts = ['171.16.11.97','171.16.11.96','171.16.11.94']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host
    client = MongoClient(conn_url)
    return client

def getMongoColl(db_name, coll_name, hosts=[]):
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host + '/' + db_name
    client = MongoClient(conn_url, minPoolSize=10, maxPoolSize=None)
    return client[db_name][coll_name]

def getNewsInHashIds(hashIds):
    news_coll = getMongoColl(db_name='nis-news', 
                           coll_name='News', 
                           hosts=['172.16.11.196', '172.16.11.195', '172.16.11.194'])
    
    newsMap = {}
    array = [t for t in news_coll.find({'_id' : {"$in" : hashIds}})]
    
    for i in range(len(array)):
        newsMap[array[i]['_id']] = array[i]
        
    return newsMap

def getNewsInDates(begin, end):
    news_coll = getMongoColl(db_name='nis-news', 
                           coll_name='News', 
                           hosts=['172.16.11.196', '172.16.11.195', '172.16.11.194'])
    
    newsMap = {}
    array = [t for t in news_coll.find({'createdAt' : {"$gt" : begin, "$lt" : end}})]
    
    for i in range(len(array)):
        newsMap[array[i]['_id']] = array[i]
        
    return newsMap


# In[3]:


import datetime
import logging
import os
import shutil
import sys
import json

import numpy as np
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import *




NIS_RAW_DATA_BASE_PATH = "gs://inshorts-segment-raw/data/segment-raw-v5/"

# fetching and processing data
conf = SparkConf().setAll([('spark.sql.broadcastTimeout',1000)
    # ('spark.driver.memory', '20g'), \
#                            ('spark.broadcast.blockSize', '10m'),\
#                            ('spark.dynamicAllocation.enabled','False'),\
#                            ('spark.executor.instances','9999')
# #                            ,("spark.sql.autoBroadcastJoinThreshold",-1)
                          ])
sc = SparkContext.getOrCreate(conf=conf)
#sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
#sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def get_path(dates, prefix='timeSpentFrontEvents', padding=None):
    date_fmt = "%Y/%m/%d"
    dates_ = dates + []
    if padding:
        st_date, ed_date = sorted(dates)[0], sorted(dates)[-1]
        for i in range(1, 4):
            d = (datetime.datetime.strptime(ed_date, date_fmt) + datetime.timedelta(days=i)).strftime(date_fmt)
            if d < datetime.datetime.today().strftime(date_fmt):
                dates_.append(d)
    dates_ = list(set(dates_) - {"2019/02/17", "2019/02/18", "2019/05/28", "2019/06/03",
                                 "2019/07/02", "2019/07/03", "2019/07/04"})
    paths = []
    for date in dates_:
        base_path = NIS_DATA_BASE_PATH
        if date < "2018/06/26":
            base_path = NIS_OLD_DATA_BASE_PATH
        paths.append(base_path + date + "/" + prefix + "/*.parquet")
    return paths

def process_parquet_data(training_dates):
    try:
        paths = get_path(training_dates)
        data = sqlContext.read.parquet(*paths)
        data = data.filter(~data.appName.isin(['mini', 'crux']))
        view_data = data.filter(data.deviceId != '')
        view_data = view_data.filter((view_data.shortTime > 1) & (view_data.shortTime < 60))
        time_udf = F.udf(lambda x: max(min(20., x), 0.) / 4 if (1 < x < 60) else -1., FloatType())
        gid_udf = F.udf(lambda x: x[:-2] if x else "", StringType())
        view_data = view_data.withColumn("timeSpent", time_udf("shortTime"))
        view_data = view_data.select(view_data.deviceId, gid_udf(view_data.hashId).alias('hashId'),
                                     view_data.timeSpent)
        view_data = view_data.groupby(view_data.deviceId, view_data.hashId)                             .agg(F.max(view_data.timeSpent).alias('timeSpent'))
        return view_data
    except Exception as e:
        logger.warning("Error processing data: " + str(e))



# In[4]:





def get_raw_path(date, hours=None):
    paths = []
    base_path = NIS_RAW_DATA_BASE_PATH + date
    if not hours:
        return base_path + "/*/*.gz"
    for hour in hours:
        paths.append(base_path + "/" + str(hour).zfill(2) + "/*.gz")
    return ",".join(paths)

def process_raw_data(paths):
    def view_data_filters(x):
        x = x['properties']
        deviceid_filter = ('deviceId' in x) and (x['deviceId'] != '')
        time_filter = ('timeSpent' in x) and (int(x['timeSpent']) <= 100)
        return deviceid_filter and time_filter

    try:
        rdd = sc.textFile(paths)             .map(json.loads)             .filter(lambda x: "batch" in x).flatMap(lambda x: x["batch"])             .filter(lambda x: ("event" in x) and (x["event"].lower() == "timespent-front"))             .filter(view_data_filters)             .map(lambda x: x['properties'])
        view_data = rdd.map(lambda x: (x['deviceId'], x['hashId'][:-2], x['timeSpent']))             .toDF(['deviceId', 'hashId', 'timeSpent'])
        view_data = view_data.groupby(view_data.deviceId, view_data.hashId)                              .agg(F.max(view_data.timeSpent).alias('timeSpent'))
        return view_data
    except Exception as e:
        logger.warning("Error processing data: " + str(e))



# In[5]:





logger = logging.getLogger(str(datetime.datetime.today().date()))
hdlr = logging.FileHandler(
    '/home/Gourav/LDA/logs/' + str(datetime.datetime.today().date()) + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def Log(s, flag=True):
    if flag:
        logger.info(s)
        print(s)

Log('----', flag=True)


# In[ ]:





SPLITS = 1

## to do
workingDate = datetime.datetime.now() - datetime.timedelta(minutes=20)
time2 = datetime.datetime.now() - datetime.timedelta(minutes=80)
time3 = datetime.datetime.now() - datetime.timedelta(minutes=140)
#workingDate = datetime.datetime(2023, 2, 18) - datetime.timedelta(minutes=20)
#time2 = datetime.datetime(2023, 2, 18) - datetime.timedelta(minutes=80)
#time3 = datetime.datetime(2023, 2, 18) - datetime.timedelta(minutes=140)

hours = [time3.hour, time2.hour, workingDate.hour]

if time2.hour == 23 and workingDate.hour == 0:
    hours = [workingDate.hour]
if time2.hour == 0 and workingDate.hour == 1:
    hours = [time2.hour, workingDate.hour]

path = get_raw_path(workingDate.strftime("%Y/%m/%d"), hours)
        
view_data = process_raw_data(path)
view_data.cache()

data_chunks = view_data.randomSplit([1.] * SPLITS)


# In[8]:


from Colab_funcn import *


NUM_DIMS = 20
NUM_CLUSTERS = 14

ALPHA = 1e-4
LAMMDA = 1e-5
CLUSTER_SHIFT_ALPHA = .9

#### Hash definition
def InternalSeqHash(deviceId):
    h = 0
    for c in list(deviceId):
        h = (31*h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000

def hashIdentifier(deviceId, M=100):
    hId = (abs(InternalSeqHash(deviceId)) % M) + 1
    return hId

### Initializer
def getDeviceVectors(deviceIds):
    
    deviceIdsSplit = np.array_split(deviceIds, 4)
    deviceVectorMap = {}
    resetCount = 0
    
    for devIds in deviceIdsSplit:
        devIdsList = list(devIds)
        deviceVectorMapProx = getDeviceVectorsFromMongo(devIdsList)
        for devId in devIdsList:
            if devId not in deviceVectorMapProx or not allVectorsOkay(deviceVectorMapProx[devId]) or len(deviceVectorMapProx[devId]) != NUM_DIMS:
                deviceVectorMap[devId] = np.random.randn(NUM_DIMS)
                resetCount += 1
            else:
                deviceVectorMap[devId] = deviceVectorMapProx[devId]

    logger.info("reset " + str(round(resetCount / len(deviceIds) * 100, 3)) + " % device vectors the db")
    
    return deviceVectorMap
        
def getNewsVectors(hashIds):
    newsVectorMap = getNewsVectorsFromMongo(hashIds)
    resetCount = 0
    for hashId in hashIds:
        """todo get it from mongo"""
        if hashId not in newsVectorMap or not allVectorsOkay(newsVectorMap[hashId]) or len(newsVectorMap[hashId]) != NUM_DIMS:
            newsVectorMap[hashId] = np.random.randn(NUM_DIMS)
            resetCount += 0
        
    logger.info("reset " + str(round(resetCount / len(hashIds) * 100, 3)) + " % news vectors the db")
    return newsVectorMap

def getNewsTSpentData(hashIds):
    newsTSpentMap = getNewsTSpentFromMongo(hashIds)
    for hashId in hashIds:
        if hashId not in newsTSpentMap:
            newsTSpentMap[hashId] = {}
        
        for cluster in range(NUM_CLUSTERS):
            if cluster not in newsTSpentMap[hashId]:
                newsTSpentMap[hashId][cluster] = (6, 1)
    
    return newsTSpentMap

def getDeviceClusters(deviceIds):
    deviceIdsSplit = np.array_split(deviceIds, 4)
    deviceClusterMap = {}
    
    for devIds in deviceIdsSplit:
        devIdsList = list(devIds)
        deviceClusterMapProx = getDeviceClustersFromMongo(devIdsList)
        for devId in devIds:
            if devId not in deviceClusterMap or deviceClusterMapProx[devId] >= NUM_CLUSTERS:
                deviceClusterMap[devId] = hashIdentifier(devId, NUM_CLUSTERS) - 1
            else:
                deviceClusterMap[devId] = deviceClusterMapProx[devId]
        
    return deviceClusterMap


# In[9]:


###data collection for logging purposes
devIdProcessMap = {}
#need to find out the hid for the dump users
#myList = ["7b5c5ea3-6da4-4c0e-9054-5ad09e2fde4b"]
#def filterDeviceId(deviceId):
#     hId = hashIdentifier(deviceId)
#     if (hId < 30 and hId > 5) or deviceId in myList:
#             return True
#    return True

def processAndTrainDf(df):
    
    newsMap = getNewsInDates(datetime.datetime.now() - datetime.timedelta(days=3), datetime.datetime.now())
    hashIdList = list(newsMap.keys())
        
    hashIdsWithFilter = []
    for h in hashIdList:
        if 'newsLanguage' in newsMap[h] and newsMap[h]['newsLanguage'] == 'english' and newsMap[h]['publishGroupList'][0]['countryCode'] == 'IN':
            hashIdsWithFilter.append(h.split('-')[0])
            
#    deviceIdsWithFilter = list(filter(filterDeviceId, df['deviceId']))

#    df = df[df['hashId'].isin(hashIdsWithFilter)]
#    df = df[df['deviceId'].isin(deviceIdsWithFilter)]
    return train(df)

def allVectorsOkay(vector):
    return np.sum(np.isnan(vector)) == 0 and np.sum(np.isinf(vector)) == 0 and np.linalg.norm(vector) < vector.shape[0] * 10
    
def train(df):
    
    deviceIds = list(df['deviceId'].unique())
    hashIds = list(df['hashId'].unique())
    
    deviceClusters = getDeviceClusters(deviceIds)
    newsTSpentMap = getNewsTSpentData(hashIds)
    
    stream = df.to_numpy()
    
    cnt_fails = 0.
    cnt_newstspent_nan = 0.
    
    for s in stream:
        deviceId = s[0]
        hashId = s[1]
        overallTimeSpent = max(min(s[2], 30), 0)
        
        if np.isnan(overallTimeSpent):
            continue
        
        deviceCluster = deviceClusters[deviceId]
        timespent, count = newsTSpentMap[hashId][deviceCluster] 
        
        if np.isnan(timespent) or timespent < 0:
            timespent = 6
            count = 1
            cnt_newstspent_nan += 1
            
        newsTSpentMap[hashId][deviceCluster] = (timespent + overallTimeSpent, count + 1)
        
        """logging"""
        if deviceId not in devIdProcessMap:
            devIdProcessMap[deviceId] = {"processed" : 0}
        
        devIdProcessMap[deviceId]["processed"] += 1
        
    if cnt_fails > 0:
        logger.warning("Got nan or inf during training %age " + str(round(cnt_fails / len(stream) * 100, 3)))
    
    if cnt_newstspent_nan > 0:
        logger.warning("Got news timespent nan " + str(round(cnt_newstspent_nan / len(stream) * 100, 3)))
    
    logger.info("processed all " + str(len(stream)) + " entries. " + " Average updates per device " + str(round(len(stream) / len(devIdProcessMap), 3)))
    
    return newsTSpentMap

def refreshNewsTSpentMap(newsTSpentMap):
    clusterSet = set(range(NUM_CLUSTERS)) 
    pass


# In[10]:


import pandas as pd

for data_df in data_chunks:
    
    data_df = data_df.toPandas()

    try:
        newsTSpentMap = processAndTrainDf(data_df)
        refreshNewsTSpentMap(newsTSpentMap)
        #insertDeviceVectorsInMongo(deviceVectors)
        #insertDeviceClustersInMongo(deviceClusters)
        #insertClusterCentersInMongo(clusterCenters)
        #insertNewsVectorsInMongo(newsVectors)
        insertNewsTSpentInMongo(newsTSpentMap)
        insertNewsScoreInMongo(newsTSpentMap)
        
        logger.info("inserted all data in the db, Num Clusters " + str(NUM_CLUSTERS))
    
    except Exception as e:
        logger.warning("Error processing data: " + str(e))


# In[11]:


Log("Exectution finished successfully, in : %s minutes, %s seconds " % (int((time.time() - start_time) / 60), int((time.time() - start_time) % 60)), flag=True)


# In[ ]:




