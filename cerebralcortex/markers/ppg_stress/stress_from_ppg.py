# Copyright (c) 2020, MD2K Center of Excellence
# All rights reserved.
#
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

from cerebralcortex.algorithms.ppg.bandpass_filter import bandpass_filter
from cerebralcortex.algorithms.ppg.cqp_quality_features_and_rr import compute_quality_features_and_rr, get_quality_likelihood
from cerebralcortex.algorithms.ppg.hrv_features import get_hrv_features, normalize_features
from cerebralcortex.core.datatypes.datastream import DataStream
import pickle

def stress_from_ppg(data:DataStream,
                    Fs:int=25,
                    low_cutoff:int=.4,
                    high_cutoff:int=3,
                    filter_order = 65,
                    rr_window_size=5.0,
                    no_of_quality_features=4,
                    ppg_columns=('red','infrared','green'),
                    acl_columns=('aclx','acly','aclz'),
                    wrist='left',
                    sensor_name='motionsensehrv',
                    quality_model_path="./model/quality_classifier.p"):
    """
    Compute stress episodes from PPG data

    Args:
        data (DataStream): dataframe of ppg & accelerometer data with column name specifying the different channels of
        sensors
        Fs (int): sampling frequency of
        low_cutoff: minimum pass band frequency
        high_cutoff: maximum pass band frequency


    Returns:
        DataStream: stress episodes computed from PPG
    """

    # High Frequency Noise Removal using bandpass filtering

    bandpass_filtered_ppg = bandpass_filter(data,Fs=Fs,low_cutoff=low_cutoff,
                                            high_cutoff=high_cutoff,filter_order=filter_order,
                                            ppg_columns=ppg_columns,acl_columns=acl_columns,
                                            wrist=wrist,sensor_name=sensor_name)

    # Compute data quality features and mean RR interval every fixed window of 5 seconds

    ppg_quality_features_rr = compute_quality_features_and_rr(bandpass_filtered_ppg,Fs=Fs,
                                                              window_size=rr_window_size,
                                                              ppg_columns=ppg_columns,
                                                              acl_columns=acl_columns,
                                                              wrist=wrist,
                                                              sensor_name=sensor_name)

    # Compute CQP data quality likelihood and remove irrecoverable segments
    quality_clf  = pickle.load(open(quality_model_path,'rb'))


    ppg_quality_likelihood = get_quality_likelihood(ppg_quality_features_rr,
                                                    clf=quality_clf,
                                                    no_of_ppg_channels=len(ppg_columns),
                                                    no_of_quality_features=no_of_quality_features,wrist=wrist,
                                                    sensor_name=sensor_name)

    ppg_stress_features = get_hrv_features(ppg_quality_likelihood,wrist=wrist,sensor_name=sensor_name)


    ppg_stress_features_normalized = normalize_features(ppg_stress_features,wrist=wrist,sensor_name=sensor_name)



    # # Normalize features
    # stress_features_normalized = normalize_features(stress_features,input_feature_array_name='features')
    #
    # # Compute stress probability
    # ecg_stress_probability = compute_stress_probability(stress_features_normalized,model_path=model_path)
    #
    # # Forward fill and impute stress data
    # ecg_stress_probability_forward_filled = forward_fill_data(ecg_stress_probability)
    # ecg_stress_probability_imputed = impute_stress_likelihood(ecg_stress_probability_forward_filled)
    #
    # # Compute stress episodes
    # stress_episodes = compute_stress_episodes(ecg_stress_probability=ecg_stress_probability_imputed)

    return stress_episodes
