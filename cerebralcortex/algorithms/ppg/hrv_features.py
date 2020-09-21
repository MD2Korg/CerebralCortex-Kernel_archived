# Copyright (c) 2020, MD2K Center of Excellence
# All rights reserved.
# Md Azim Ullah (mullah@memphis.edu)
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
from scipy import signal
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StructType, DoubleType, StringType, TimestampType, IntegerType, ArrayType
import numpy as np
import pandas as pd
from cerebralcortex.algorithms.utils.mprov_helper import CC_MProvAgg
from cerebralcortex.core.datatypes import DataStream
from cerebralcortex.core.metadata_manager.stream.metadata import Metadata, DataDescriptor, \
    ModuleMetadata
from copy import deepcopy
from scipy.stats import iqr
from pyspark.sql import functions as F

def frequencyDomain(RRints,
                    tmStamps,
                    vlf= (0.003, 0.04),
                    lf = (0.04, 0.15),
                    hf = (0.15, 0.4)):
    """
    Calculates Frequency domain features

    :param RRints: RR intervals
    :param tmStamps: Timestamps
    :param vlf: Very low frequency range
    :param lf: low frequency range
    :param hf: High frequency range
    :return: a dictionery of 4 frequency domain features
    """
    NNs = RRints
    tss = tmStamps
    frequency_range = np.linspace(0.001, 1, 10000)
    NNs = np.array(NNs)
    NNs = NNs - np.mean(NNs)
    result = signal.lombscargle(tss, NNs, frequency_range)

    #Pwelch w/ zero pad
    fxx = frequency_range
    pxx = result

    df = fxx[1] - fxx[0]
    vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx = df)
    lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx = df)
    hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx = df)
    totalPower = vlf_power + lf_power + hf_power

    #Normalize and take log
    vlf_NU_log = np.log((vlf_power / (totalPower - vlf_power)) + 1)
    lf_NU_log = np.log((lf_power / (totalPower - vlf_power)) + 1)
    hf_NU_log = np.log((hf_power / (totalPower - vlf_power)) + 1)
    lfhfRation_log = np.log((lf_power / hf_power) + 1)

    freqDomainFeats = {'VLF_Power': vlf_NU_log, 'LF_Power': lf_NU_log,
                       'HF_Power': hf_NU_log, 'LF/HF': lfhfRation_log}

    return freqDomainFeats

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    :param RR int values
    :param Weights
    :return average and variance weighted
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, variance

def get_weighted_rr_features(a,ts,step=3):
    """
    computes hrv features from list of RR intervals
    :param a: n*2 shape, first column is the RR ints and 2nd column contains the likelihoods
    :param ts: Timestamp array
    :param step: weighting step
    :return: numpy ndarray containing the features
    """
    m,s = weighted_avg_and_std(deepcopy(a[:,0]), deepcopy(a[:,1]))
    for i in range(a.shape[0]):
        if i<step:
            a[i,0] = np.average(a[:i+1,0],weights=a[:i+1,1]+1e-6)
        else:
            a[i,0] = np.average(a[i-step:i+1,0],weights=a[i-step:i+1,1]+1e-6)
    p75 = np.percentile(a[:,0],75)
    p25 = np.percentile(a[:,0],25)
    p50 = np.percentile(a[:,0],50)
    p80 = np.percentile(a[:,0],80)
    p20 = np.percentile(a[:,0],20)
    feature_freq = frequencyDomain(a[:,0]/1000,ts/1000)
    return np.array([np.var(a[:,0]),p75-p25,
                     np.std(a[:,0]),np.sqrt(np.mean(np.diff(a[:,0])** 2)),
                     m,p50,p80,p20,
                     60000/p50]+list(feature_freq.values()))

def get_metadata(data,
                 wrist='left',
                 sensor_name='motionsensehrv'):
    """
    :param data: input stream
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor

    :return: metadata of output stream
    """
    stream_name = "org.md2k."+str(sensor_name)+"."+str(wrist)+".stress.features.minute"
    stream_metadata = Metadata()
    stream_metadata.set_name(stream_name).set_description("Minute level stress features calculated from PPG rr intervals per minute") \
        .add_input_stream(data.metadata.get_name()) \
        .add_dataDescriptor(DataDescriptor().set_name("min").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("mean").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("median").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("start").set_type("timestamp")) \
        .add_dataDescriptor(DataDescriptor().set_name("end").set_type("timestamp")) \
        .add_dataDescriptor(DataDescriptor().set_name("timestamp").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("localtime").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("version").set_type("int")) \
        .add_dataDescriptor(DataDescriptor().set_name("user").set_type("string")) \
        .add_dataDescriptor(DataDescriptor().set_name("features").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("day").set_type("string")) \
        .add_dataDescriptor(DataDescriptor().set_name("activity_values").set_type("array"))

    stream_metadata.add_module(
        ModuleMetadata().set_name("Minute level stress features calculated from PPG rr intervals per minute")
            .set_attribute("url", "http://md2k.org/")
            .set_author("Md Azim Ullah", "mullah@memphis.edu"))

    return stream_metadata


def get_hrv_features(data,
                     acceptable_rr_ints_per_minute=0.5,
                     no_of_rr_ints_per_minute=30,
                     not_wearing_standard_deviation_threshold=0.001174897554939529,
                     mean_quality_threshold=0.2,
                     minimum_rr=400,
                     maximum_rr=1200,
                     wrist='left',
                     sensor_name='motionsensehrv'):
    """

    :param data: input data
    :param acceptable_rr_ints_per_minute: minimum acceptable rr interval ratio
    :param no_of_rr_ints_per_minute: maximum no. of rr interval datapoints possible per minute depends on the sliding window size
    :param not_wearing_standard_deviation_threshold: accelerometer standard deviation signifying sensor wearing
    :param mean_quality_threshold: signal quality threshold for a minute
    :param minimum_rr: minimum RR interval
    :param maximum_rr: maximum RR interval
    :param wrist: writ on which sensor is worn
    :param sensor_name: name of sensor
    :return: datastream containing minute level stress features computed from PPG based RR intervals
    """


    schema = StructType([
        StructField('min',DoubleType()),
        StructField('median',DoubleType()),
        StructField('mean',DoubleType()),
        StructField("start", TimestampType()),
        StructField("end", TimestampType()),
        StructField("features", ArrayType(DoubleType())),
        StructField("user", StringType()),
        StructField('version',DoubleType()),
        StructField("timestamp", TimestampType()),
        StructField("localtime", TimestampType()),
        StructField("activity_values", ArrayType(DoubleType()))
    ])
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def get_minute_data(key,df):
        if df.shape[0]<no_of_rr_ints_per_minute*acceptable_rr_ints_per_minute or df.activity.std()<not_wearing_standard_deviation_threshold:
            return pd.DataFrame([],columns=['min','mean','median','start','end','features',
                                            'user','version','localtime','timestamp',"activity_values"])

        df['likelihood_max'] = df['likelihood_max'].apply(lambda a:a if a<1 else a-.01)

        if df.likelihood_max.mean()<mean_quality_threshold:
            return pd.DataFrame([],columns=['min','mean','median','start','end','features',
                                            'user','version','localtime','timestamp',"activity_values"])

        df['time'] = df['time'].apply(lambda a:a*1000)
        df = df.dropna().sort_values('start').reset_index(drop=True)

        ecg = df[['rr','likelihood_max','time']].values.reshape(-1,3)
        ecg = ecg[(ecg[:,0]>=minimum_rr)&(ecg[:,0]<=maximum_rr),:]
        times = ecg[:,2]
        if ecg.shape[0]<no_of_rr_ints_per_minute*acceptable_rr_ints_per_minute:
            return pd.DataFrame([],columns=['min','mean','median','start','end','features',
                                            'user','version','localtime','timestamp',"activity_values"])
        ecg_features = get_weighted_rr_features(ecg[:,:2],times)
        temp = []
        temp.append([df['likelihood_max'].min(),
                     df['likelihood_max'].mean(),
                     np.median(df['likelihood_max']),
                     key[2]['start'],
                     key[2]['end'],
                     ecg_features,
                     df.user.values[0],
                     df.version.values[0],
                     df.localtime.values[0],
                     df.timestamp.values[0],
                     np.array(list(df['activity']))])
        return pd.DataFrame(temp,columns=['min','mean','median','start','end','features',
                                          'user','version','localtime','timestamp',"activity_values"])

    data = data.withColumn('time',F.col('start').cast('double'))
    win = F.window("timestamp", windowDuration='60 seconds', startTime='0 seconds')
    groupbycols = ["user","version"] + [win]
    data_final = data._data.groupBy(groupbycols).apply(get_minute_data)
    data_final = data_final.withColumn('day',F.date_format('localtime',"YYYYMMdd"))
    metadata = get_metadata(data,wrist=wrist,sensor_name=sensor_name)
    return DataStream(data=data_final,metadata=metadata)


def normalize_features(data,
                       index_of_first_order_feature =4,
                       lower_percentile=20,
                       higher_percentile=99,
                       minimum_minutes_in_day=60,
                       maximum_acceptable_motion=0.2,
                       no_features=13,
                       epsilon = 1e-8,
                       input_feature_array_name='features',
                       wrist='left',
                       sensor_name='motionsensehrv'):
    """

    :param data: input datastream
    :param index_of_first_order_feature: index of feature column containing mean RR interval
    :param lower_percentile: lower percentile number of rr intervals
    :param higher_percentile: higher percentile number of rr intervals
    :param minimum_minutes_in_day: minutes per day to be present
    :param maximum_acceptable_motion: maximum acceptable motion per minute to be included in normalization
    :param no_features: no of features in the features column
    :param epsilon: epsilon value to avoid zero division
    :param input_feature_array_name: name of features column
    :return: normalized features
    """

    data_day = data.withColumn('day',F.date_format('localtime','yyyyMMdd'))
    stream_metadata = data.metadata
    stream_metadata \
        .add_input_stream(data.metadata.get_name()) \
        .add_dataDescriptor(
        DataDescriptor()
            .set_name("features_normalized")
            .set_type("array")
            .set_attribute("description","All features normalized daywise"))
    data_day = data_day.withColumn('features_normalized',F.col(input_feature_array_name))

    schema = data_day._data.schema
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def normalize_features(data):
        if len(data)<minimum_minutes_in_day:
            return pd.DataFrame([], columns=data.columns)

        quals1 = np.array([1] * data.shape[0])
        feature_matrix = np.array(list(data[input_feature_array_name])).reshape(-1, no_features)
        ss = np.repeat(feature_matrix[:,index_of_first_order_feature],np.int64(np.round(100*quals1)))
        rr_70th = np.percentile(ss,lower_percentile)
        rr_95th = np.percentile(ss,higher_percentile)
        activity_75th = np.array([np.percentile(a,75) for a in data['activity_values'].values])
        index = np.where((feature_matrix[:,index_of_first_order_feature]>rr_70th)&
                         (feature_matrix[:,index_of_first_order_feature]<rr_95th)&
                         (activity_75th<=maximum_acceptable_motion))[0]
        for i in range(feature_matrix.shape[1]):
            m,s = weighted_avg_and_std(feature_matrix[index,i], quals1[index])
            s+=epsilon
            feature_matrix[:,i]  = (feature_matrix[:,i] - m)/s
        data['features_normalized']  = list([np.array(b) for b in feature_matrix])
        return data

    data_normalized = data_day._data.groupby(['user','day','version']).apply(normalize_features)
    features = DataStream(data=data_normalized,metadata=stream_metadata)
    return features
