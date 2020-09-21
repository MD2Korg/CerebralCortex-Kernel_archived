# Copyright (c) 2020, MD2K Center of Excellence
# All rights reserved.
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

from scipy.stats import skew,kurtosis
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy import signal
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StructType, DoubleType, StringType, TimestampType, IntegerType, ArrayType
import numpy as np
import pandas as pd
from cerebralcortex.algorithms.utils.mprov_helper import CC_MProvAgg
from cerebralcortex.core.datatypes import DataStream
from cerebralcortex.core.metadata_manager.stream.metadata import Metadata, DataDescriptor, \
    ModuleMetadata



### Peak Detection codes ##

def _datacheck_peakdetect(x_axis, y_axis):
    """
    check data for peak detection

    :param x_axis: time
    :param y_axis: values
    :return: same as input data
    """
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """

    :param y_axis: values
    :param x_axis: time
    :param lookahead: steps ahead to look for
    :param delta:
    :return: peak locations

    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break

    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        pass

    return [max_peaks, min_peaks]

### CQP quality features and heart rate estimation

def get_predict_prob(window,
                     Fs=25):

    """
    Get CQP quality features
    :param window: Numpy array of PPG data
    :param Fs: Sampling Frequency
    :return: quality features
    """
    no_channels = window.shape[1]
    window[:,:] = signal.detrend(RobustScaler().fit_transform(window),axis=0)
    f,pxx = signal.welch(window,fs=Fs,nperseg=len(window),nfft=10000,axis=0)
    pxx = np.abs(pxx)
    pxx = MinMaxScaler().fit_transform(pxx)
    skews = skew(window,axis=0).reshape(no_channels,1)
    kurs = kurtosis(window,axis=0).reshape(no_channels,1)
    iqrs = np.std(window,axis=0).reshape(no_channels,1)
    rps = np.divide(np.trapz(pxx[np.where((f>=.8)&(f<=2.5))[0]],axis=0),np.trapz(pxx,axis=0)).reshape(no_channels,1)
    features = np.concatenate([skews,kurs,rps,iqrs],axis=1)
    return features

def get_rr_value(values,Fs=25,nfft=10000):
    """
    Get Mean RR interval

    :param values: single channel ppg data
    :param Fs: sampling frequency
    :param nfft: FFT no. of points
    :return: Mean RR interval Information
    """
    try:
        f, pxx = signal.welch(values,fs=Fs,nperseg=values.shape[0],nfft=nfft,axis=0)
        f = f.reshape(-1)
        pxx = pxx.reshape(-1,1)
        peakind =  peakdetect(pxx[:,0],lookahead=2)
        x = []
        y = []
        for a in peakind[0]:
            x.append(a[0])
            y.append(a[1])
        x = np.array(x)
        x = x[f[x]>.8]
        x = x[f[x]<2.5]
        f = f[x]
        pxx = pxx[x,0]
        return 60000/(60*f[np.argmax(pxx)])
    except Exception as e:
        return 0


def get_rr_and_features(window,
                        Fs=25):
    """

    :param window: Data
    :param Fs: Sampling Frequency
    :return: tuple of mean RR interval and Quality features calculated
    """
    no_channels = window.shape[1]
    no_of_features = 4
    starts = [0]
    rrs = []
    features= []
    for i,s in enumerate(starts):
        for j in range(window.shape[1]):
            rrs.append(get_rr_value(window[s:,j],Fs=Fs))
        features.append(get_predict_prob(window[s:,:],Fs=Fs).reshape(1,no_channels,no_of_features))
    return np.array(rrs),np.concatenate(features).reshape(no_channels,no_of_features)



def get_metadata_features_rr(data,
                 wrist='left',
                 sensor_name='motionsensehrv'):
    """
    :param data: input stream
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor

    :return: metadata of output stream
    """
    stream_name = "org.md2k."+str(sensor_name)+"."+str(wrist)+".wrist.features.activity.std.rr"
    stream_metadata = Metadata()
    stream_metadata.set_name(stream_name).set_description("PPG data quality features and mean RR interval computed from fixed window") \
        .add_input_stream(data.metadata.get_name()) \
        .add_dataDescriptor(DataDescriptor().set_name("timestamp").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("localtime").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("version").set_type("int")) \
        .add_dataDescriptor(DataDescriptor().set_name("user").set_type("string")) \
        .add_dataDescriptor(DataDescriptor().set_name("features").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("rr").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("activity").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("start").set_type("timestamp")) \
        .add_dataDescriptor(DataDescriptor().set_name("end").set_type("timestamp"))

    stream_metadata.add_module(
        ModuleMetadata().set_name("PPG data quality features and  mean RR Interval computed from PPG")
            .set_attribute("url", "http://md2k.org/")
            .set_author("Md Azim Ullah", "mullah@memphis.edu"))
    return stream_metadata


def compute_quality_features_and_rr(data,
                                    Fs=25,
                                    window_size=5.0,
                                    acceptable_percentage=0.8,
                                    ppg_columns=['red','infrared','green'],
                                    acl_columns=['aclx','acly','aclz'],
                                    wrist='left',
                                    sensor_name='motionsensehrv'):
    """

    :param data: Input data
    :param Fs: Sampling Frequency
    :param window_size: Window size to compute features from
    :param acceptable_percentage: minimum acceptable data fraction
    :param ppg_columns: columns in input data belonging to PPG
    :param acl_columns: columns in input data belonging to Accelerometer
    :param wrist: wrist on which the sensor was worn
    :param sensor_name: name of sensor
    :return: Dataframe containing PPG data quality features and mean RR interval information
    """

    ## check if all columns exist

    default_columns = ['user','version','localtime','timestamp']
    required_columns = default_columns+list(acl_columns)+list(ppg_columns)
    if len(set(required_columns)-set(data.columns))>0:
        raise Exception("Columns missing in input dataframe! " + str(list(set(required_columns)-set(data.columns))))

    ## select the columns from input dataframe

    data = data.select(*required_columns)

    ## udf
    default_schema = [StructField("timestamp", TimestampType()),
                      StructField("localtime", TimestampType()),
                      StructField("version", IntegerType()),
                      StructField("user", StringType())]
    output_schema = [StructField("features", ArrayType(DoubleType())),
                     StructField("rr", ArrayType(DoubleType())),
                     StructField("activity", DoubleType()),
                     StructField("start", TimestampType()),
                     StructField("end", TimestampType())]
    schema = StructType(default_schema+output_schema)
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ppg_features_compute(key,data):
        if data.shape[0]>window_size*Fs*acceptable_percentage:
            data = data.sort_values('timestamp').reset_index(drop=True)
            rows = []
            rows.append(data['user'].loc[0])
            rows.append(data['version'].loc[0])
            rows.append(data['timestamp'].loc[0])
            rows.append(data['localtime'].loc[0])
            rrs , features = get_rr_and_features(data[list(ppg_columns)].values.reshape(-1,len(ppg_columns)),Fs=Fs)
            rows.append(rrs)
            rows.append(features.reshape(-1))
            data_acl = data[list(acl_columns)]
            values_acl = data_acl.values
            acl_std = np.std(values_acl,axis=0)
            acl_std = np.sqrt(np.sum(np.square(acl_std)))
            rows.append(acl_std)
            rows.append(key[2]['start'])
            rows.append(key[2]['end'])
            return pd.DataFrame([rows],columns=['user','version',
                                                'timestamp','localtime',
                                                'rr','features','activity',
                                                'start','end'])

        else:
            return pd.DataFrame([],columns=['user','version',
                                            'timestamp','localtime',
                                            'rr','features','activity',
                                            'start','end'])

    ppg_features_and_rr = data.compute(ppg_features_compute,windowDuration=window_size,slideDuration=int(window_size//2),startTime='0 seconds')
    output_data = ppg_features_and_rr._data
    ds = DataStream(data=output_data,metadata=get_metadata_features_rr(data,wrist=wrist,sensor_name=sensor_name))

    return ds



def get_metadata_likelihood(data,
                 wrist='left',
                 sensor_name='motionsensehrv'):
    """
    :param data: input stream
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor

    :return: metadata of output stream
    """
    stream_name = "org.md2k."+str(sensor_name)+"."+str(wrist)+".wrist.likelihood.activity.std.rr"
    stream_metadata = Metadata()
    stream_metadata.set_name(stream_name).set_description("PPG data quality likelihood, channel selection and rr interval") \
        .add_input_stream(data.metadata.get_name()) \
        .add_dataDescriptor(DataDescriptor().set_name("timestamp").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("localtime").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("version").set_type("int")) \
        .add_dataDescriptor(DataDescriptor().set_name("user").set_type("string")) \
        .add_dataDescriptor(DataDescriptor().set_name("features").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("rr").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("rr_array").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("likelihood_max").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("likelihood_max_array").set_type("array")) \
        .add_dataDescriptor(DataDescriptor().set_name("activity").set_type("double")) \
        .add_dataDescriptor(DataDescriptor().set_name("start").set_type("timestamp")) \
        .add_dataDescriptor(DataDescriptor().set_name("end").set_type("timestamp"))

    stream_metadata.add_module(
        ModuleMetadata().set_name("PPG data quality features and  mean RR Interval computed from PPG")
            .set_attribute("url", "http://md2k.org/")
            .set_author("Md Azim Ullah", "mullah@memphis.edu"))
    return stream_metadata





def get_quality_likelihood(data,
                           clf,
                           no_of_ppg_channels = 3,
                           no_of_quality_features = 4,
                           wrist='left',
                           sensor_name='motionsensehrv'):
    """

    :param data: input data
    :param clf: classifier for quality classification
    :param no_of_ppg_channels: no of PPG channels
    :param no_of_quality_features: No. of features in PPG quality estimation
    :param wrist: wrist on which sensor is worn
    :param sensor_name: name of sensor
    :return: Datastream with
    """
    if 'features' not in data.columns or 'rr' not in data.columns:
        raise Exception("Required columns not present, Please fix it!")


    ## helper method
    def convert_to_array(vals):
        return np.array(vals).reshape(no_of_ppg_channels,no_of_quality_features)

    ## udf
    schema = StructType([
        StructField("version", IntegerType()),
        StructField("user", StringType()),
        StructField("localtime", TimestampType()),
        StructField("timestamp", TimestampType()),
        StructField("likelihood_max", DoubleType()),
        StructField("rr", DoubleType()),
        StructField("likelihood_max_array", ArrayType(DoubleType())),
        StructField("rr_array", ArrayType(DoubleType())),
        StructField("activity", DoubleType()),
        StructField("start", TimestampType()),
        StructField("end", TimestampType()),
        StructField("features", ArrayType(DoubleType())),
    ])
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ppg_likelihood_compute(data):
        if data.shape[0]>0:
            data['features'] = data['features'].apply(convert_to_array)
            acl_features = np.concatenate(list(data['features'])).reshape(-1, no_of_ppg_channels,no_of_quality_features)
            likelihood = []
            for k in range(acl_features.shape[1]):
                tmp = np.nan_to_num(acl_features[:,k,:]).reshape(-1,no_of_quality_features)
                likelihood.append(clf.predict_proba(tmp)[:,1].reshape(-1,1))

            likelihood = np.concatenate(likelihood,axis=1)
            rrs = data['rr'].values
            rrs = np.array([np.array(a) for a in rrs])
            likelihood_max = []
            rr = []
            rr_array = []
            likelihood_max_array = []
            for i in range(likelihood.shape[0]):
                a = likelihood[i,:]
                likelihood_max_array.append(list(a))
                rr_array.append(list(rrs[i]))
                likelihood_max.append(np.max(a))
                rr.append(rrs[i][np.argmax(a)])
            data['likelihood_max'] = likelihood_max
            data['rr'] = rr
            data['likelihood_max_array'] = likelihood_max_array
            data['rr_array'] = rr_array
            data['features'] = data['features'].apply(lambda a:a.reshape(-1))
            return data
        else:
            return pd.DataFrame([],columns=['user','version','timestamp','localtime','likelihood_max',
                                            'rr','activity','likelihood_max_array','rr_array','start','end','features'])

    ppg_likelihood = data._data.groupBy(['user','version']).apply(ppg_likelihood_compute)
    metadata = get_metadata_likelihood(data,wrist=wrist,sensor_name=sensor_name)
    return DataStream(data=ppg_likelihood,metadata=metadata)




