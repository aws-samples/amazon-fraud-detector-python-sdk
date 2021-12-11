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

import uuid
import json


import logging
from datetime import datetime
import time
from IPython.display import clear_output, JSON

import boto3
import numpy as np
import pandas as pd

lh = logging.getLogger('frauddetector')

class FraudDetector:
    """FraudDetector class to build, train and deploy.

    This class helps to build, train and deploy a Amazon Fraud Detector
    project. It implements the three most common methods for model deployment:
    # - .fit()
    # - .deploy()
    # - .predict()

    """

    def __init__(self, entity_type, event_type, model_name, model_version, model_type,
                 detector_name, detector_version="1"):
        """Build, train and deploy Amazon Fraud Detector models.

        Technical documentation on how Amazon Fraud Detector works can be
        found at: https://docs.aws.amazon.com/frauddetector/


        Args:
            :entity_type:          represents who is performing the event
            :event_type:           defines the structure for an individual event
            :model_name:           name of model to be created or used
            :model_version:        model version
            :model_type:           ONLINE_FRAUD_INSIGHTS / TRANSACTION_FRAUD_INSIGHTS
            :detector_name:        name for the fraud detection project
            :detector_version:     versioning for fraud detection
            :variables:            AWS Fraud Detector list of JSON variable defs
            :labels:               AWS Fraud Detector list of JSON label defs

        """
        # super(FraudDetector, self).__init__()
        self.fd = boto3.client("frauddetector")
        self.s3 = boto3.client("s3")
        self.iam = boto3.client('iam')
        self.entity_type = entity_type
        self.event_type = event_type
        self.detector_name = detector_name
        self.detector_version = detector_version
        self.model_name = model_name
        if "." not in str(model_version):  # check if missing decimal point - if so append ".00"
            self.model_version = model_version + ".00"
        else:
            self.model_version = model_version
        self.model_type = model_type
        #Initialize empty variables
        self.project_variables = None
        self.project_labels = None
        self.variables = None
        self.labels = None
        self.events = None
        self.entities = None
        self.models = None

    @staticmethod
    def get_entity_types(self):
        """Get entities already created in Amazon Fraud Detector cloud service"""
        self.entities = self.fd.get_entity_types()
    
    @staticmethod
    def get_event_types(self):
        """Get events already created in Amazon Fraud Detector cloud service"""
        self.events = self.fd.get_event_types()

    @staticmethod
    def get_variables(self):
        """Get variables already created in Amazon Fraud Detector cloud service"""
        self.variables = self.fd.get_variables()
        
    @staticmethod
    def get_labels(self):
        """Get labels already created in Amazon Fraud Detector cloud service"""
        self.variables = self.fd.get_labels()
        
    @staticmethod
    def get_models(self):
        """Get models already created in Amazon Fraud Detector cloud service"""
        self.models = self.fd.get_models()
    
    def _setup_project(self):
        """Automatically setup your Amazon Fraud Detector project."""
        response = self.create_entity_type()
        response = self.create_labels(labels=self.project_labels)
        response = self.create_variables(variables=self.project_variables)
        response = self.create_event_type(variables=self.project_variables, labels=self.project_labels)
        response = self.create_model()
        return "Success"
    
    def create_model(self):
        """Create Amazon FraudDetector model. Wraps the boto3 SDK API to allow bulk operations.
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/frauddetector.html#FraudDetector.Client.create_model

        Args:
            None
            
        Returns:
            :response_all:      {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        existing_names = [m['modelId'] for m in self.models['models']]
        response_all = []

        if self.model_name not in existing_names:

            lh.debug("create_model: {}".format(self.model_name))
            # create event via Boto3 SDK fd instance
            response = self.fd.create_model(
                eventTypeName=self.event_type,
                modelId=self.model_name,
                modelType=self.model_type
            )
            lh.info("create_model: entity {} created".format(self.model_name))
            status = {self.model_name: response['ResponseMetadata']['HTTPStatusCode']}
            response_all.append(status)
        else:
            lh.warning("create_model: entity {} already exists, skipping".format(self.model_name))
            status = {self.model_name: "skipped"}
            response_all.append(status)

        # update myself
        self.models = self.fd.get_models()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all

    def set_model_version_inactive(self):
        # set model inactive
        response = self.fd.update_model_version_status(
            modelId=self.model_name,
            modelType=self.model_type,
            modelVersionNumber=self.model_version,
            status='INACTIVE'
        )
        return response
    
    def delete_model(self):
        """Delete Amazon FraudDetector event. Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :event:          name of the event to delete

        Returns:
            :response:
        """

        # delete model
        response = self.fd.delete_model(
            modelId=self.model_name,
            modelType=self.model_type
        )
        lh.info("delete_model: model {} deleted".format(self.model_name,self.model_version))
        status = {self.model_name: response['ResponseMetadata']['HTTPStatusCode']}

        # update myself
        self.models = self.fd.get_models()

        return status
        
    def create_entity_type(self):
        """Create Amazon FraudDetector entity. Wraps the boto3 SDK API to allow bulk operations.
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/frauddetector.html#FraudDetector.Client.put_entity_type

        Args:
            None
            
        Returns:
            :response_all:      {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        existing_names = [e['name'] for e in self.entities['entityTypes']]
        response_all = []

        if self.entity_type not in existing_names:

            lh.debug("create_entity_type: {}".format(self.entity_type))
            # create event via Boto3 SDK fd instance
            response = self.fd.put_entity_type(
                name = self.entity_type
            )
            lh.info("create_entity_type: entity {} created".format(self.entity_type))
            status = {self.entity_type: response['ResponseMetadata']['HTTPStatusCode']}
            response_all.append(status)
        else:
            lh.warning("create_entity_type: entity {} already exists, skipping".format(self.entity_type))
            status = {self.event_type: "skipped"}
            response_all.append(status)

        # update myself
        self.entities = self.fd.get_entity_types()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all
    
    def delete_entity_type(self):
        """Delete Amazon FraudDetector event. Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :event:          name of the event to delete

        Returns:
            :response_all:   {variable_name: API-response-status, variable_name: API-response-status} dict
        """
        response_all = []
        response = self.fd.delete_entity_type(
            name=self.entity_type,
        )
        lh.info("delete_entity_type: entity {} deleted".format(self.entity_type))
        status = {self.entity_type: response['ResponseMetadata']['HTTPStatusCode']}
        response_all.append(status)

        # update myself
        self.entities = self.fd.get_entity_types()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all
        
    def create_event_type(self, variables, labels):
        """Create Amazon FraudDetector event. Wraps the boto3 SDK API to allow bulk operations.
        https://docs.aws.amazon.com/frauddetector/latest/ug/create-event-type.html
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/frauddetector.html#FraudDetector.Client.put_event_type

        Args:
            :variables: (list)  List object containing variables to instantiate with associated Amazon Fraud Detector types
                                [
                                    {
                                        "name": "email_address",
                                        "variableType": "EMAIL_ADDRESS",
                                        "dataType": "STRING",
                                        "defaultValue": "unknown",
                                        "description": "email address",
                                        "tags": [{"VariableName": "email_address"}, }]
                                    },
                                    ...
                                ]
            :labels: (list)    List object containing labels to instantiate with associated Amazon Fraud Detector
                                        [
                                            {
                                                "name": "legit"
                                            },
                                            {
                                                "name": "fraud"
                                            }
                                        ]
            
        Returns:
            :response_all:      {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        existing_names = [e['name'] for e in self.events['eventTypes']]
        response_all = []

        if self.event_type not in existing_names:

            lh.debug("create_event_type: {}".format(self.event_type))
            # create event via Boto3 SDK fd instance
            response = self.fd.put_event_type(
                name = self.event_type,
                eventVariables = [v["name"] for v in variables],
                labels = [l["name"] for l in labels],
                entityTypes = [self.entity_type]
            )
            lh.info("create_event_type: event {} created".format(self.event_type))
            status = {self.event_type: response['ResponseMetadata']['HTTPStatusCode']}
            response_all.append(status)
        else:
            lh.warning("create_event_type: event {} already exists, skipping".format(self.event_type))
            status = {self.event_type: "skipped"}
            response_all.append(status)

        # update myself
        self.events = self.fd.get_event_types()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all
    
    def delete_event_type(self):
        """Delete Amazon FraudDetector event. Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :event:          name of the event to delete

        Returns:
            :response_all:   {variable_name: API-response-status, variable_name: API-response-status} dict
        """
        response_all = []
        lh.info("delete_event_type: event {}".format(self.event_type))
        response = self.fd.delete_event_type(
            name=self.event_type,
        )

        status = {self.event_type: response['ResponseMetadata']['HTTPStatusCode']}
        response_all.append(status)

        # update myself
        self.events = self.fd.get_event_types()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all

    def create_variables(self, variables):
        """Create Amazon FraudDetector variables.  Wraps the boto3 SDK API to allow bulk operations.
        https://docs.aws.amazon.com/frauddetector/latest/ug/create-a-variable.html
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/frauddetector.html#FraudDetector.Client.create_variable

        Args:
            :variables: (list)  List object containing variables to instantiate with associated Amazon Fraud Detector types
                                [
                                    {
                                        "name": "email_address",
                                        "variableType": "EMAIL_ADDRESS",
                                        "dataType": "STRING",
                                        "defaultValue": "unknown",
                                        "description": "email address",
                                        "tags": [{"VariableName": "email_address"}, }]
                                    },
                                    ...
                                ]
        Returns:
            :response_all:      {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        existing_names = [v['name'] for v in self.fd.get_variables()['variables']]
        response_all = []

        for v in variables:
            if v['name'] not in existing_names:

                # handle missing keys for incomplete JSON spec
                data_type = v.get('dataType')
                default_value = v.get('defaultValue')
                if default_value is None:
                    if v['variableType'] != "NUMERIC":
                        default_value = '<unknown>'
                    else:
                        default_value = "0.0"
                # create variables via Boto3 SDK fd instance
                lh.debug("create_variables: {} {} defaultValue {}".format(v['name'], v['variableType'], default_value))
                response = self.fd.create_variable(
                    name=v['name'],
                    variableType=v['variableType'],
                    dataSource='EVENT',
                    dataType=data_type,
                    defaultValue=default_value
                    )
                lh.info("create_variables: variable {} created".format(v['name']))
                status = {v['name']: response['ResponseMetadata']['HTTPStatusCode']}
                response_all.append(status)
            else:
                lh.warning("create_variables: variable {} already exists, skipping".format(v['name']))
                status = {v['name']: "skipped"}
                response_all.append(status)

        # update myself
        self.variables = self.fd.get_variables()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all

    def delete_variables(self, variables):
        """Delete Amazon FraudDetector variables.  Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :variables:      list of variable-names to delete

        Returns:
            :response_all:   {variable_name: API-response-status, variable_name: API-response-status} dict
        """
        response_all = []
        for vname in variables:
            response = self.fd.delete_variable(
                name=vname,
            )
            lh.info("delete_variables: variable {} deleted".format(vname))
            status = {vname: response['ResponseMetadata']['HTTPStatusCode']}
            response_all.append(status)

        # update myself
        self.variables = self.fd.get_variables()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all
    
    def create_labels(self, labels):
        """Create Amazon FraudDetector labels.  Wraps the boto3 SDK API to allow bulk operations.
        https://docs.aws.amazon.com/frauddetector/latest/ug/create-a-label.html
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/frauddetector.html#FraudDetector.Client.put_label

        Args:
            :labels: (list)    List object containing labels to instantiate with associated Amazon Fraud Detector
                                        [
                                            {
                                                "name": "legit"
                                            },
                                            {
                                                "name": "fraud"
                                            }
                                        ]
        Returns:
            :response_all:              {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        existing_names = [l['name'] for l in self.labels['labels']]
        response_all = []

        for l in labels:
            if l['name'] not in existing_names:
                # create label via Boto3 SDK fd instance
                lh.debug("put_label: {}".format(l['name']))
                response = self.fd.put_label(
                    name=l['name'],
                    description=l['name']
                )
                lh.info("create_labels: label {} created".format(l['name']))
                status = {l['name']: response['ResponseMetadata']['HTTPStatusCode']}
                response_all.append(status)
            else:
                lh.warning("create_labels: label {} already exists, skipping".format(l['name']))
                status = {l['name']: "skipped"}
                response_all.append(status)

        # update myself
        self.variables = self.fd.get_labels()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all

    def delete_labels(self, labels):
        """Delete Amazon FraudDetector labels. Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :labels:      list of label-names to delete

        Returns:
            :response_all:   {variable_name: API-response-status, variable_name: API-response-status} dict
        """
        response_all = []
        for lname in labels:
            response = self.fd.delete_label(
                name=lname,
            )
            lh.info("delete_labels: label {} deleted".format(lname))
            status = {lname: response['ResponseMetadata']['HTTPStatusCode']}
            response_all.append(status)

        # update myself
        self.variables = self.fd.get_labels()

        # convert list of dicts to single dict
        response_all = {k: v for d in response_all for k, v in d.items()}
        return response_all

    def create_outcomes(self, outcomes_list):
        """ Create outcomes for detector
            Args:
                :outcomes_list:          list; list of (outcome_name, outcome_description) tuples
        """
        for outcome in outcomes_list:
            self.fd.put_outcome(
                name=outcome[0],
                description=outcome[1]
            )

    def delete_outcomes(self, outcomes_list):
        """ Delete outcomes for detector
            Args:
                :outcomes_list:          list; list of (outcome_name, outcome_description) tuples
        """
        for outcome in outcomes_list:
            self.fd.delete_outcome(
                name=outcome[0]
            )

    @property
    def outcomes(self):
        outcomes_response = self.fd.get_outcomes()['outcomes']
        names = [x['name'] for x in outcomes_response]
        descriptions = [x['description'] for x in outcomes_response]

        return list(zip(names, descriptions))

    def fit(self, data_schema, data_location, role, variables, labels, data_source="EXTERNAL_EVENTS", wait=False):
        """Train Amazon FraudDetector model version. Wraps the boto3 SDK API to allow bulk operations.

        Args:
            :wait:      boolean to indicate whether to wait or not

        Returns:
            :response_all:   {variable_name: API-response-status, variable_name: API-response-status} dict
        """

        self.project_variables = variables
        self.project_labels = labels
        self.variables = self.fd.get_variables()
        self.labels = self.fd.get_labels()
        self.events = self.fd.get_event_types()
        self.entities = self.fd.get_entity_types()
        self.models = self.fd.get_models()
        if self.variables and self.labels:
            self._setup_project()


        event_details = {
            'dataLocation'     : data_location,
            'dataAccessRoleArn': role
        }

        lh.info("fit: train {} model".format(self.model_name))
        response = self.fd.create_model_version(
            modelId=self.model_name,
            modelType=self.model_type,
            trainingDataSource=data_source,
            trainingDataSchema=data_schema,
            externalEventsDetail=event_details
        )
        lh.info("Wait for model training to complete...")
        stime = time.time()
        while wait:
            current_time = datetime.now()
            clear_output(wait=True)
            response = self.fd.get_model_version(
                modelId=self.model_name,
                modelType=self.model_type,
                modelVersionNumber=self.model_version)
            if response['status'] == 'TRAINING_IN_PROGRESS':
                lh.info(f"{current_time}: current progress: {(time.time() - stime)/60:{3}.{3}} minutes")
                time.sleep(60)  # -- sleep for 60 seconds 
            if response['status'] != 'TRAINING_IN_PROGRESS':
                lh.info(f"{current_time}: Model status : {response['status']}")
                break
        etime = time.time()

        # -- summarize -- 
        lh.info("\nModel training complete")
        lh.info("\nElapsed time : %s" % (etime - stime) + " seconds \n")
        return response

    @property
    def model_status(self):
        model_version_number = str(self.model_version)
        # check if missing decimal point - if so append ".00" to work around format requirement in FD
        if "." not in model_version_number:
            model_version_number = model_version_number + ".00"

        response = self.fd.get_model_version(modelId=self.model_name,
                                         modelType=self.model_type,
                                         modelVersionNumber=model_version_number)
        return response['status']

    def activate(self, outcomes_list=None):
        """create a Fraud Detector detector and activate a model with outcomes
        Args:

            :outcomes_list:          list; list of (outcome_name, outcome_description) tuples
        """

        if self.model_status != 'TRAINING_COMPLETE' and self.model_status != 'ACTIVE':
            raise EnvironmentError("model training must be complete before compiling")

        # create a new detector
        self.fd.put_detector(
            detectorId=self.detector_name,
            eventTypeName=self.event_type
        )

        # put outcomes - if no outcomes, skip this (may be activating a model to work with existing outcomes)
        if outcomes_list:
            self.create_outcomes(outcomes_list=outcomes_list)

        # Activate the model
        self.fd.update_model_version_status(
            modelId=self.model_name,
            modelType=self.model_type,
            modelVersionNumber=self.model_version,
            status='ACTIVE'
        )

    def delete_detector_version(self):
        """Deletes the detector-version associated with this instance"""
        response = self.fd.delete_detector_version(
            detectorId=self.detector_name,
            detectorVersionId=self.detector_version
        )
        return response

    def predict(self, event_timestamp, event_variables, entity_id="unknown"):
        """Predict using your Amazon Forecast model

        Args:
            :event_timestamp:   A string indicating the timestamp key
            :event_variables:   A dict with your event variables
            :entity_id:         The unique ID of your entity if known

        Returns:
            :score:   {'credit_card_model_insightscore': 14.0, 'ruleResults': ['verify_outcome']} dict
        """
        response = self.fd.get_event_prediction(
            detectorId=self.detector_name,
            detectorVersionId=self.detector_version,
            eventId=str(uuid.uuid4()),
            eventTypeName=self.event_type,
            entities=[
                {
                    'entityType': self.entity_type,
                    'entityId': entity_id
                },
            ],
            eventTimestamp=event_timestamp,
            eventVariables = event_variables
        )
        score = response['modelScores'][0]["scores"]
        score["ruleResults"] = response['ruleResults']
        return score

    def batch_predict(self, timestamp, events=None, df=None, entity_id="unknown"):
        """Batch predict using your Amazon Forecast model

        Args:
            :timestamp:   A string indicating either the timestamp key or column
            :events:      A list of JSON events
            :df:          A Pandas DataFrame with your observations for prediction
            :entity_id:   The unique ID of your entity if known

        Returns:
            :predictions:   [{'credit_card_model_insightscore': 14.0, 'ruleResults': ['verify_outcome']}] list
        """
        if events is None and df is None:
            print("Please provide either a JSON object through events or a Pandas DataFrame through df!")
            return []
        predictions = []
        if type(events) == "dict":
            for event in events:
                event_timestamp = event[timestamp]
                tmp = event.pop(timestamp)
                predictions.append(
                    self.predict(
                        event_timestamp=event_timestamp,
                        event_variables=event,
                        entity_id=entity_id))
        else:
            try:
                events.loc[:, timestamp] = events.loc[:, timestamp].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%dT%H:%M:%SZ'))
                for i in range(events.shape[0]):
                    event = json.loads(events.iloc[i, :].to_json())
                    for key in event:
                        event[key] = str(event[key])
                    event_timestamp = event[timestamp]
                    tmp = event.pop(timestamp)
                    predictions.append(
                        self.predict(
                            event_timestamp=event_timestamp,
                            event_variables=event,
                            entity_id=entity_id))
            except Exception as e:
                print("Warning: Make sure your input DataFrame complies with the service rules!")
                print(e)
        return predictions

    @property
    def rules(self):
        """list of rules associated with this instance's detector
                paginated API not used - max limit 100 records
                """
        rules = self.fd.get_rules(
            detectorId=self.detector_name
        )
        return rules['ruleDetails']

    def create_rules(self, rules):
        """Create rules by passing in a list of dictionaries with expressions and outcomes they map to
        Args:

            :rules:          list; list of dictionaries
                            [{'ruleId': 'name_of_rule',
                                'expression': 'rule_expression_for_evaluating_rule',
                                'outcomes': [list_of, outcomes_for, matching_rule]
                                },
                             {'ruleId': 'name-of-next-rule'...
                            ]
        https://docs.aws.amazon.com/frauddetector/latest/ug/rule-language-reference.html
        """
        # ToDo Checks: rules map to existing outcomes
        responses = []
        try:
            existing_rules = [r['ruleId'] for r in self.rules]
        except KeyError:
            lh.info("create_rules: No pre-existing rules found")
        for rule in rules:
            if rule['ruleId'] not in existing_rules:
                response = self.fd.create_rule(
                    ruleId=rule['ruleId'],
                    detectorId=self.detector_name,
                    description="Rule: " + rule['ruleId'] + " for outcomes " + str(rule['outcomes']),
                    expression=rule['expression'],
                    outcomes=rule['outcomes'],
                    language="DETECTORPL"
                )
                responses.append(response)
            else:
                lh.warning("create_rules: rule {} already exists, skipping".format(rule['ruleId']))
                responses.append("skipped")

        return responses

    def delete_rules(self, rules):
        """delete a list of rules (cannot delete a rule if it is used by an ACTIVE or INACTIVE detector version)
        Args:

            :rules:
        """

        for r in rules:
            try:
                response = self.fd.delete_rule(
                    rule={'detectorId': self.detector_name, 'ruleId': r['ruleId'], 'ruleVersion': r['ruleVersion']}
                )
            except Exception as e:
                lh.warning("delete_rules: " + str(e))

    def deploy(self, rules_list=None, rule_execution_mode='FIRST_MATCHED'):
        """Deploy a detector-version with associated rules for a particular model version

        Args:

            :rules_list:  Optional: if a list of rules is supplied, call the create_rules method, otherwise work with existing rules
                     pass in list of (rule_name, expression, [outcomes]) tuples

        """

        # Check model-version is ACTIVE
        if self.model_status != 'ACTIVE':
            lh.warning("deploy: Model is not active; wait until model is active before deploying")
            raise EnvironmentError("Model not active")

        # create rules, if supplied, then get all the rules associated with this detector instance
        if rules_list:
            response = self.create_rules(rules_list)
        active_rules = self.rules

        # create a rules list of dicts to pass in to create detector version
        rules_payload = []
        for r in active_rules:
            rules_payload.append({'detectorId': self.detector_name, 'ruleId': r['ruleId'], 'ruleVersion': r['ruleVersion'] })

        # deploy the detector-version with rules and model
        response = self.fd.create_detector_version(
            detectorId=self.detector_name,
            rules=rules_payload,
            modelVersions=[{
                'modelId': self.model_name,
                'modelType': self.model_type,
                'modelVersionNumber': self.model_version
                }],
            ruleExecutionMode=rule_execution_mode
        )
        return response

    def delete_detector(self):
        response = self.fd.delete_detector_version(
            detectorId=self.detector_name,
            detectorVersionId=self.detector_version
        )
        return response
