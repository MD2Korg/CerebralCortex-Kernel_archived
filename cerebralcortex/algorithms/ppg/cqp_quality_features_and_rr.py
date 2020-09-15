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

def get_predict_prob(window):
    """
    Get CQP quality features
    :param window: Numpy array of PPG data
    :return: quality features
    """
    no_channels = window.shape[1]
    window[:,:] = signal.detrend(RobustScaler().fit_transform(window),axis=0)
    f,pxx = signal.welch(window,fs=25,nperseg=len(window),nfft=10000,axis=0)
    pxx = np.abs(pxx)
    pxx = MinMaxScaler().fit_transform(pxx)
    skews = skew(window,axis=0).reshape(no_channels,1)
    kurs = kurtosis(window,axis=0).reshape(no_channels,1)
    iqrs = np.std(window,axis=0).reshape(no_channels,1)
    rps = np.divide(np.trapz(pxx[np.where((f>=.8)&(f<=2.5))[0]],axis=0),np.trapz(pxx,axis=0)).reshape(no_channels,1)
    features = np.concatenate([skews,kurs,rps,iqrs],axis=1)
    return features

def get_rr_value(values,fs=25):
    """
    Get Mean RR interval

    :param values: single channel ppg data
    :param fs: sampling frequency
    :return: Mean RR interval Information
    """
    try:
        f, pxx = signal.welch(values,fs=fs,nperseg=values.shape[0],nfft=10000,axis=0)
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


def get_rr_and_features(window):
    """

    :param window:
    :return: tuple of mean RR interval and Quality features calculated
    """
    no_channels = window.shape[1]
    starts = [0]
    ends = [125]
    rrs = []
    features= []
    for i,s in enumerate(starts):
        e = ends[i]
        for j in range(window.shape[1]):
            rrs.append(get_rr_value(window[s:,j]))
        features.append(get_predict_prob(window[s:e,:]).reshape(1,no_channels,4))
    return np.array(rrs),np.concatenate(features).reshape(no_channels,4)



def get_metadata(data,
                 wrist='left',
                 sensor_name='motionsensehrv'):
    """
    :param data: input stream
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor

    :return: metadata of output stream
    """
    stream_name = "org.md2k."+str(sensor_name)+"."+str(wrist)+".wrist.features.activity.std"
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
                                    ppg_columns=('red','infrared','green'),
                                    acl_columns=('aclx','acly','aclz'),
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
            rrs , features = get_rr_and_features(data[list(ppg_columns)].values.reshape(-1,len(ppg_columns)))
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

    ppg_features_and_rr = data.compute(ppg_features_compute,windowDuration=5,slideDuration=1,startTime='0 seconds')
    output_data = ppg_features_and_rr._data
    ds = DataStream(data=output_data,metadata=get_metadata(data,wrist=wrist,sensor_name=sensor_name))

    return ds



