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

import json

import boto3
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class Metrics:
    """Return metrics for a given Amazon Fraud Detector project.

    Attributes:
    project_name    The name of the Amazon Fraud Detector project.
    fd              The Amazon Fraud Detector boto3 client.

    """

    def __init__(self, project_name):
        """Creates a Metrics object for Amazon Fraud Detector.
        It can retrieve key metrics, namely precision, recall and f1 score
        from a given Amazon Fraud Detector project.

        Technical documentation on how Amazon Fraud Detector works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            project_name (str): Name of the Amazon Fraud Detector to interact with.

        """
        super(Metrics, self).__init__()
        self.project_name = project_name
        self.fd = boto3.client("frauddetector")
        self.s3 = boto3.client("s3")
