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

import pandas as pd
from pandas._testing import assert_frame_equal
import pytest

from frauddetector import profiler

EVENT_COLUMN = "EVENT_LABEL"
TIMESTAMP_COLUMN = "EVENT_TIMESTAMP"
LABELS = [{'name': 'legit'}, {'name': 'fraud'}]
VARS = [{
            'name': 'Category',
            'variableType': 'CATEGORY',
            'dataType': 'STRING',
            'defaultValue': 'unknown'
        },
        {
            'name': 'Value',
            'variableType': 'NUMERIC',
            'dataType': 'FLOAT',
            'defaultValue': 'unknown'
        }
    ]

DATA = pd.DataFrame(
    data=[
        ["A", 42, "legit", "21-07-2021 11:01:23"],
        ["B", 24, "fraud", "21-07-2021 12:05:13"],
        ["B", 42, "legit", "21-07-2021 03:50:43"],
        ["C", 42, "legit", "21-07-2021 01:36:06"]],
    columns=["Category", "Value", EVENT_COLUMN, TIMESTAMP_COLUMN])


SUMMARY = pd.DataFrame(
    data=[
        ['Category', "object", 4, 3, 0, 4, 0.0, 0.75],
        ['Value', "int64", 4, 2, 0, 4, 0.0, 0.5],
        ['EVENT_LABEL', "object", 4, 2, 0, 4, 0.0, 0.5],
        ['EVENT_TIMESTAMP', "object", 4, 4, 0, 4, 0.0, 1.0]],
    columns=["feature_name", "dtype", "count", "nunique", "null", "not_null", "null_pct", "nunique_pct"])

def test___calculate_summary_stats():
    prof = profiler.Profiler()
    stats = prof._Profiler__calculate_summary_stats(data=DATA)
    assert_frame_equal(stats, SUMMARY)
    
def test___check_column_in_dataframe():
    prof = profiler.Profiler()
    column_in_df = prof._Profiler__check_column_in_dataframe(data=DATA)
    assert column_in_df == True
    
def test___map_feature_types():
    prof = profiler.Profiler()
    SUMMARY["feature_type"] = ['CATEGORY', 'NUMERIC', 'TARGET', 'EVENT_TIMESTAMP']
    stats = prof._Profiler__calculate_summary_stats(data=DATA)
    maps = prof._Profiler__map_feature_types(df_stats=stats)
    assert_frame_equal(maps, SUMMARY)
    
def test___screen_for_warnings():
    prof = profiler.Profiler()
    SUMMARY["feature_type"] = ['CATEGORY', 'NUMERIC', 'TARGET', 'EVENT_TIMESTAMP']
    SUMMARY["feature_warning"] = ['NO WARNING', 'NO WARNING', 'NO WARNING', 'NO WARNING']
    stats = prof._Profiler__calculate_summary_stats(data=DATA)
    maps = prof._Profiler__map_feature_types(df_stats=stats)
    warns = prof._Profiler__screen_for_warnings(df_types=maps)
    assert_frame_equal(warns, SUMMARY)
    
def test_get_summary_stats_table():
    prof = profiler.Profiler()
    SUMMARY["feature_type"] = ['CATEGORY', 'NUMERIC', 'TARGET', 'EVENT_TIMESTAMP']
    SUMMARY["feature_warning"] = ['NO WARNING', 'NO WARNING', 'NO WARNING', 'NO WARNING']
    stats = prof.get_summary_stats_table(data=DATA)
    assert_frame_equal(stats, SUMMARY)

def test__create_labels():
    prof = profiler.Profiler()
    DATA[EVENT_COLUMN] = [x.lower() for x in DATA[EVENT_COLUMN].tolist()]
    labels = prof._Profiler__create_labels(data=DATA, event_column=EVENT_COLUMN)
    assert labels == LABELS

def test___create_variables():
    prof = profiler.Profiler()
    stats = prof.get_summary_stats_table(data=DATA)
    variables = prof._Profiler__create_variables(df_stats=stats, event_column=EVENT_COLUMN, timestamp_column=TIMESTAMP_COLUMN)
    assert variables == VARS

def test___extract_frauddetector_schema():
    prof = profiler.Profiler()
    SUMMARY["feature_type"] = ['CATEGORY', 'NUMERIC', 'TARGET', 'EVENT_TIMESTAMP']
    SUMMARY["feature_warning"] = ['NO WARNING', 'NO WARNING', 'NO WARNING', 'NO WARNING']
    warns = prof.get_summary_stats_table(data=DATA)
    DATA[EVENT_COLUMN] = [x.lower() for x in DATA[EVENT_COLUMN].tolist()]
    data_schema, variables, labels = prof._Profiler__extract_frauddetector_schema(
        data=DATA,
        df_warn=warns,
        event_column=EVENT_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        filter_warnings=False)
    assert data_schema == {
        'modelVariables': ['Category', 'Value'],
        'labelSchema': {
            'labelMapper': {
                'FRAUD': ['fraud'],
                'LEGIT': ['legit']
            }
        }
    }
    assert variables == VARS
    assert labels == LABELS

def test_get_frauddetector_inputs():
    prof = profiler.Profiler()
    DATA[EVENT_COLUMN] = [x.lower() for x in DATA[EVENT_COLUMN].tolist()]
    SUMMARY["feature_type"] = ['CATEGORY', 'NUMERIC', 'TARGET', 'EVENT_TIMESTAMP']
    SUMMARY["feature_warning"] = ['NO WARNING', 'NO WARNING', 'NO WARNING', 'NO WARNING']
    data_schema, variables, labels = prof.get_frauddetector_inputs(
        data=DATA,
        event_column=EVENT_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        filter_warnings=False)
    assert data_schema == {
        'modelVariables': ['Category', 'Value'],
        'labelSchema': {
            'labelMapper': {
                'FRAUD': ['fraud'],
                'LEGIT': ['legit']
            }
        }
    }
    assert variables == VARS
    assert labels == LABELS