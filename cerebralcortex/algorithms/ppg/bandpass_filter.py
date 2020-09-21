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

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructField, StructType, DoubleType, StringType, TimestampType, IntegerType
import numpy as np
from cerebralcortex.algorithms.utils.mprov_helper import CC_MProvAgg
from cerebralcortex.core.datatypes import DataStream
from cerebralcortex.core.metadata_manager.stream.metadata import Metadata, DataDescriptor, \
    ModuleMetadata


from scipy import signal
def filter_data(X,
                Fs=100,
                low_cutoff=.4,
                high_cutoff=3.0,
                filter_order=65):
    """
    Bandpass Filter of single channel

    :param X: input data
    :param Fs: sampling freq.
    :param low_cutoff: low passband
    :param high_cutoff: high passband
    :param filter_order: no of taps in FIR filter

    :return: filtered version of input data
    """
    X1 = X.reshape(-1,1)
    X1 = signal.detrend(X1,axis=0,type='constant')
    b = signal.firls(filter_order,np.array([0,low_cutoff-.1, low_cutoff, high_cutoff ,high_cutoff+.5,Fs/2]),np.array([0, 0 ,1 ,1 ,0, 0]),
                     np.array([100*0.02,0.02,0.02]),fs=Fs)
    X2 = signal.convolve(X1.reshape(-1),b,mode='same')
    return X2

def get_metadata(data,
                 wrist='left',
                 sensor_name='motionsensehrv',
                 ppg_columns=('red','infrared','green'),
                 acl_columns=('aclx','acly','aclz')):
    """
    :param data: input stream
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor
    :param ppg_columns: columns in the input dataframe referring to multiple ppg channels
    :param acl_columns: columns in the input dataframe referring to accelerometer channels

    :return: metadata of output stream
    """
    stream_name = "org.md2k."+str(sensor_name)+"."+str(wrist)+".wrist.bandpass.filtered"
    stream_metadata = Metadata()
    stream_metadata.set_name(stream_name).set_description("Bandpass Filtered PPG data") \
        .add_input_stream(data.metadata.get_name()) \
        .add_dataDescriptor(DataDescriptor().set_name("timestamp").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("localtime").set_type("datetime")) \
        .add_dataDescriptor(DataDescriptor().set_name("version").set_type("int")) \
        .add_dataDescriptor(DataDescriptor().set_name("user").set_type("string"))

    for c in ppg_columns:
        stream_metadata.add_dataDescriptor(DataDescriptor().set_name(c).set_type("double").set_attribute("description",
                                                                                                    "ppg channel "+c))
    for c in acl_columns:
        stream_metadata.add_dataDescriptor(DataDescriptor().set_name(c).set_type("double").set_attribute("description",
                                                                                            "accelerometer channel "+c))

    stream_metadata.add_module(
        ModuleMetadata().set_name("ecg data quality").set_attribute("url", "http://md2k.org/").set_author(
            "Md Azim Ullah", "mullah@memphis.edu"))
    return stream_metadata


def bandpass_filter(
                   data,
                   Fs = 25,
                   low_cutoff = 0.4,
                   high_cutoff = 3.0,
                   filter_order = 65,
                   ppg_columns=['red','infrared','green'],
                   acl_columns=['aclx','acly','aclz'],
                   wrist='left',
                   sensor_name='motionsensehrv'):

    """

    :param data: PPG & ACL data stream
    :param Fs: sampling frequency
    :param low_cutoff: minimum frequency of pass band
    :param high_cutoff: Maximum Frequency of pass band
    :param filter_order: no. of taps in FIR filter
    :param ppg_columns: columns in the input dataframe referring to multiple ppg channels
    :param acl_columns: columns in the input dataframe referring to accelerometer channels
    :param wrist: which wrist the data was collected from
    :param sensor_name: name of sensor

    :return: Bandpass filtered version of input PPG data
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
    schema = StructType(default_schema+[StructField(c, DoubleType()) for c in list(ppg_columns)+list(acl_columns)])
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def ppg_bandpass(data):
        data = data.sort_values('timestamp').reset_index(drop=True)
        for c in ppg_columns:
            data[c] = filter_data(data[c].values,Fs=Fs,low_cutoff=low_cutoff,high_cutoff=high_cutoff,filter_order=filter_order)
        return data

    ## steps
    ppg_bandpass_filtered = data.compute(ppg_bandpass,windowDuration=60*60*10,startTime='0 seconds')
    output_data = ppg_bandpass_filtered._data
    ds = DataStream(data=output_data,metadata=get_metadata(data,wrist=wrist,sensor_name=sensor_name,
                                                           ppg_columns=ppg_columns,acl_columns=acl_columns))
    return ds








