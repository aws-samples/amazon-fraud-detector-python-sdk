# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import inspect
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import boto3
import pandas as pd
import numpy as np

class Profiler:
    """Profiler class to build, train and deploy.

    This class helps to build, train and deploy a Amazon Fraud Detector
    project. It implements the three most common methods for model deployment:
    # - .summary_stats()

    Attributes:
    s3    The s3 boto3 client

    """

    def __init__(self):
        """Build, train and deploy Amazon Fraud Detector models.

        Technical documentation on how Amazon Fraud Detector works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            None

        """
        self.s3 = boto3.client("s3")

    def __calculate_summary_stats(self, data, event_column="EVENT_LABEL"):
        """ Generate summary statistics for a panda's data frame 
            
            Args:
                data (pandas.core.frame.DataFrame): panda's dataframe to create summary statistics for
                event_column (str): column that contains the target event
            Returns:
                df_stats (pandas.core.frame.DataFrame): DataFrame of summary statistics, training data schema, event variables and event lables
        """
        df = data.copy("deep")
        rowcnt = len(df)
        df[event_column] = df[event_column].astype('str', errors='ignore')
        df_s1  = df.agg(['count', 'nunique']).transpose().reset_index().rename(columns={"index":"feature_name"})
        df_s1["null"] = (rowcnt - df_s1["count"]).astype('int64')
        df_s1["not_null"] = rowcnt - df_s1["null"]
        df_s1["null_pct"] = df_s1["null"] / rowcnt
        df_s1["nunique_pct"] = df_s1['nunique']/ rowcnt
        dt = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index":"feature_name", 0:"dtype"})
        df_stats = pd.merge(dt, df_s1, on='feature_name', how='inner').round(4)
        df_stats['nunique'] = df_stats['nunique'].astype('int64')
        df_stats['count'] = df_stats['count'].astype('int64')
        return df_stats
    
    def __map_feature_types(self, df_stats):
        """Map features types in the stats table.
            
            Args:
                df_stats (pandas.core.frame.DataFrame): DataFrame of summary statistics, training data schema, event variables and event lables
            Returns:
                df_types (pandas.core.frame.DataFrame): DataFrame with mapping
        """
        df_types = df_stats.copy("deep")
        df_types['feature_type'] = "UNKOWN"
        df_types.loc[df_types["dtype"] == object, 'feature_type'] = "CATEGORY"
        df_types.loc[(df_types["dtype"] == "int64") | (df_types["dtype"] == "float64"), 'feature_type'] = "NUMERIC"
        df_types.loc[df_types["feature_name"].str.contains("ipaddress|ip_address|ipaddr"), 'feature_type'] = "IP_ADDRESS"
        df_types.loc[df_types["feature_name"].str.contains("email|email_address|emailaddr"), 'feature_type'] = "EMAIL_ADDRESS"
        df_types.loc[df_types["feature_name"] == "EVENT_LABEL", 'feature_type'] = "TARGET"
        df_types.loc[df_types["feature_name"] == "EVENT_TIMESTAMP", 'feature_type'] = "EVENT_TIMESTAMP"
        return df_types
    
    def __screen_for_warnings(self, df_types):
        """Screen over type mappings for warnings.
            
            Args:
                df_types (pandas.core.frame.DataFrame): DataFrame with mapping
            Returns:
                df_warn (pandas.core.frame.DataFrame): DataFrame with added warnings
        """
        df_warn = df_types.copy("deep")
        df_warn['feature_warning'] = "NO WARNING"
        df_warn.loc[(df_warn["nunique"] != 2) & (df_warn["feature_name"] == "EVENT_LABEL"),'feature_warning' ] = "LABEL WARNING, NON-BINARY EVENT LABEL"
        df_warn.loc[(df_warn["nunique_pct"] > 0.9) & (df_warn['feature_type'] == "CATEGORY") ,'feature_warning' ] = "EXCLUDE, GT 90% UNIQUE"
        df_warn.loc[(df_warn["null_pct"] > 0.2) & (df_warn["null_pct"] <= 0.5), 'feature_warning' ] = "NULL WARNING, GT 20% MISSING"
        df_warn.loc[df_warn["null_pct"] > 0.5,'feature_warning' ] = "EXCLUDE, GT 50% MISSING"
        df_warn.loc[((df_warn['dtype'] == "int64" ) | (df_warn['dtype'] == "float64" ) ) & (df_warn['nunique_pct'] < 0.2), 'feature_warning' ] = "LIKELY CATEGORICAL, NUMERIC w. LOW CARDINALITY"
        return df_warn

    def __create_labels(self, data, event_column):
        """Create target labels for AFD
            
            Args:
                data (pandas.core.frame.DataFrame): panda's dataframe to create summary statistics for
                event_column (str): column that contains the target event
            Returns:
                label_list (list): List of dicts with label names
        """
        if len(data[event_column].unique()) > 2:
            logging.error(f"Target column {event_column} has more than 2 unique values!")
            return None
        labels = data[event_column].unique().tolist()
        label_list = [{"name": x} for x in labels]
        return label_list

    def __create_variables(self, df_stats, event_column, timestamp_column):
        """Create variables for AFD
            
            Args:
                df_stats (pandas.core.frame.DataFrame): DataFrame of summary statistics, training data schema, event variables and event lables
                event_column (str): column that contains the target event
            Returns:
                variables (list): List of dicts with variable names
        """
        variables = []
        for i in range(df_stats.shape[0]):
            if df_stats.loc[i, "feature_name"] not in [event_column, timestamp_column]:
                data_type = "STRING"
                default_value = "unknown"
                if df_stats.loc[i, "feature_type"] == "NUMERIC":
                    data_type = "FLOAT"
                    default_value = 0.0
                variables.append({
                    "name": str(df_stats.loc[i, "feature_name"]),
                    "variableType": df_stats.loc[i, "feature_type"],
                    "dataType": data_type,
                    "defaultValue": "unknown"
                })
        return variables
    
    def get_summary_stats_table(self, data, event_column="EVENT_LABEL", timestamp_column="EVENT_TIMESTAMP"):
        """Get a summary stats table with variable warnings
            
            Args:
                data (pandas.core.frame.DataFrame): panda's dataframe to create summary statistics for
                event_column (str): column that contains the target event
                timestamp_column (str): column that contains the timestamp
            Returns:
                df (pandas.core.frame.DataFrame): DataFrame of summary statistics, training data schema, event variables and event lables
        """
        df = data.copy("deep")
        df = self.__calculate_summary_stats(data, event_column=event_column)
        df = self.__map_feature_types(df_stats=df)
        df = self.__screen_for_warnings(df_types=df)
        return df
    
    def __extract_frauddetector_schema(self, data, df_warn, event_column="EVENT_LABEL", timestamp_column="EVENT_TIMESTAMP", filter_warnings=False):
        """Get the Amazon Fraud Detector inputs:
            * training data schema
            * event_variables
            * event_labels
            
            Args:
                data (pandas.core.frame.DataFrame): panda's dataframe to create summary statistics for
                df_warn (pandas.core.frame.DataFrame): DataFrame with added warnings
                event_column (str): column that contains the target event
                timestamp_column (str): column that contains the timestamp
                filter_warning (bool): Flag for filtering out warnings
            Returns:
                data_schema (dict): The training data schema for AFD
        """
        df = df_warn.copy("deep")
        if filter_warnings:
            df = df[(df_stats['feature_warning'] != 'NO WARNING')]
        variables = self.__create_variables(df_stats=df, event_column=event_column, timestamp_column=timestamp_column)
        labels = self.__create_labels(data=data, event_column=event_column)

        data_schema = {
            'modelVariables' : df.loc[(df['feature_type'].isin(['IP_ADDRESS', 'EMAIL_ADDRESS', 'CATEGORY', 'NUMERIC']))]['feature_name'].to_list(),
            'labelSchema'    : {
                'labelMapper' : {
                    'FRAUD' : [data[event_column].value_counts().idxmin()],
                    'LEGIT' : [data[event_column].value_counts().idxmax()]
                }
            }
        }
        return data_schema, variables, labels
    
    def get_frauddetector_inputs(self, data, event_column="EVENT_LABEL", timestamp_column="EVENT_TIMESTAMP", filter_warnings=False):
        """Get the Amazon Fraud Detector inputs:
            * training data schema
            * event_variables
            * event_labels
            
            Args:
                data (pandas.core.frame.DataFrame): panda's dataframe to create summary statistics for
                df_warn (pandas.core.frame.DataFrame): DataFrame with added warnings
                event_column (str): column that contains the target event
                timestamp_column (str): column that contains the timestamp
                filter_warning (bool): Flag for filtering out warnings
            Returns:
                data_schema (dict): The training data schema for AFD
        """
        df = data.copy("deep")
        df_warn = self.get_summary_stats_table(data=df, event_column=event_column)
        return self.__extract_frauddetector_schema(
            data=data,
            df_warn=df_warn,
            event_column=event_column,
            timestamp_column=timestamp_column,
            filter_warnings=filter_warnings)