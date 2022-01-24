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
import logging
import pytest

from frauddetector import frauddetector

lh = logging.getLogger('test_frauddetector')

MODEL_VERSION = "1"
VARIABLES = [
            {
                "name": "test_email_address",
                "variableType": "EMAIL_ADDRESS",
                "dataType": "STRING",
                "defaultValue": "unknown",
                "description": "email address",
                "tags": [{"VariableName": "email_address"}, {"VariableType": "EMAIL_ADDRESS"}]    
            },
            {
                "name": "test_ip_address",
                "variableType": "IP_ADDRESS",
                "dataType": "STRING"
            },
            {
                "name": "test_quantity",
                "variableType": "NUMERIC",
                "dataType": "FLOAT"
            },
            {
                "name": "test_widget_class",
                "variableType": "CATEGORICAL",
                "dataType": "STRING"
            }
        ]

LABELS = [
    {
        "name": "test_legit"
    },
    {
        "name": "test_fraud"
    }
]

DATA = [
    ("my.name@fake.com", "192.168.0.254", 45, "A", "test_fraud"),
    ("a.fake@bla.com", "172.168.10.1", 45, "B", "test_legit"),
    ("my.name@something.com", "82.24.61.99", 32, "B", "test_legit"),
    ("my.name@another.com", "72.12.11.1", 12, "C", "test_legit"),
    ("fred.flintsone@xyz.com", "192.168.0.1", 11, "B", "test_fraud"),
    ("dtrump@abc.com", "10.0.16.21", 99, "A", "test_fraud"),
]


class TestFraudDetectorClass:

    @classmethod
    def setup_class(cls):
        lh.debug("class setup: {}".format(cls.__name__))
        cls.fd = frauddetector.FraudDetector(model_version=MODEL_VERSION,
                                             entity_type="test_transaction",
                                             event_type="test_credit_card_transaction",
                                             model_name="test_credit_card_model",
                                             detector_name="test_credit_card_fraud_project",
                                             model_type="ONLINE_FRAUD_INSIGHTS",
                                             region='eu-west-1',
                                             #variables=VARIABLES,
                                             #labels=LABELS
                                             )

    @classmethod
    def teardown_class(cls):
        lh.debug("class teardown: {}".format(cls.__name__))
        cls.fd.delete_model()
        cls.fd.delete_event_type()
        cls.fd.delete_entity_type()
        response = cls.fd.delete_variables([v['name'] for v in VARIABLES])
        response = cls.fd.delete_labels([l['name'] for l in LABELS])

    #def setup_method(self, method):
    #    lh.info("method setup: {}".format(method.__name__))

    #def teardown_method(self, method):
    #    lh.info("method teardown: {}".format(method.__name__))

    def test_aws_connect(self):
        response = self.fd.iam.list_account_aliases()
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200

    # test project with variables, labels, model and event type is created when FraudDetector instance is instantiated
    def test_fraud_project(self):
        # add variables
        self.fd.create_variables(VARIABLES)

        # test that the FraudDetector instance has had variables attribute updated
        variable_names = [v['name'] for v in self.fd.variables['variables']]
        # confirm variables exist as subset of all variables that exist (others may exist outside the test framework)
        assert (set(variable_names)).issuperset(
            {"test_ip_address", "test_email_address", "test_quantity", "test_widget_class"})

        # remove variables
        status = self.fd.delete_model()
        #print(status)
        self.fd.delete_event_type()
        variables_list = [v['name'] for v in VARIABLES]
        response = self.fd.delete_variables(variables_list)

        # test that the FraudDetector instance has had variables attribute updated
        variable_names = [v['name'] for v in self.fd.variables['variables']]
        # confirm test variables no longer exist as subset of all variables that exist
        for v in variable_names:
            assert v not in set({"test_ip_address", "test_email_address", "test_quantity", "test_widget_class"})

    def test_fraud_outcomes(self):
        test_outcomes = [("test_outcome1", "this is test outcome 1"), ("test_outcome2", "this is test outcome 2")]

        self.fd.create_outcomes(test_outcomes)
        outcomes = self.fd.outcomes
        assert (set(outcomes)).issuperset(
            test_outcomes)

        self.fd.delete_outcomes(test_outcomes)
        outcomes = self.fd.outcomes
        assert (set(test_outcomes)) not in set(outcomes)

    @pytest.mark.skip(reason="can only run this if the AWS environment and pre-created model is available")
    def test_rules(self):
        """
        Test creating rules and outcomes for a pre-existing ACTIVE model called
            registration_model (Version 1.0)
        that is associated with Detector
            registration-project (Version 1)

        Test dependency on pre-creating the detector and model in /example/frauddetector_sdk_example.ipynb
        """

        test_outcomes = [("test_outcome1_b", "this is test outcome 1"), ("test_outcome2_b", "this is test outcome 2")]

        test_rules = [{'ruleId': 'test_rule1',
                       'expression': '$registration_model_insightscore > 900',
                       'outcomes': ["test_outcome1_b"]
                      },
                      {'ruleId': 'test_rule2',
                       'expression': '$registration_model_insightscore <= 900',
                       'outcomes': ["test_outcome1_b", "test_outcome2_b"]
                      }
                     ]

        detector = frauddetector.FraudDetector(
            entity_type="transaction",
            event_type="registrations",
            detector_name="registration-project",
            model_name="registration_model",
            model_version="1.0",
            model_type="ONLINE_FRAUD_INSIGHTS",
            region='eu-west-1',
            detector_version="1"
        )

        # create outcomes to map the rules to
        detector.create_outcomes(test_outcomes)

        detector.create_rules(test_rules)

        rules = detector.rules
        rule_ids = [r['ruleId'] for r in rules]
        assert "test_rule1" in rule_ids

        # clean up rules
        live_test_rules = [r for r in rules if r['ruleId'] in ['test_rule1', 'test_rule2']]
        detector.delete_rules(live_test_rules)

        # clean-up test outcomes
        detector.delete_outcomes(test_outcomes)

        assert "test_rule1" not in [r['ruleId'] for r in detector.rules]

    @pytest.mark.skip(reason="can only run this if the AWS environment and pre-created model is available")
    def test_predict(self):
        """Test predictions for a pre-existing ACTIVE model called
                    registration_model (Version 1.0)
            that is associated with Detector
                    registration-project (Version 1)
            Rules and outcomes should be in place in preparation as well:
                    rules: high_fraud_risk, low_fraud_risk, no_fraud_risk
                    outcomes: approve_outcome, review_outcome, verify_outcome

            Test has dependency on pre-creating the detector and model using /example/frauddetector_sdk_example.ipynb
        """

        detector = frauddetector.FraudDetector(
            entity_type="registration",
            event_type="user-registration",
            detector_name="registration-project",
            model_name="registration_model",
            model_version="1.0",
            model_type="ONLINE_FRAUD_INSIGHTS",
            region='eu-west-1',
            detector_version="1"
        )

        event_variables = {
            'email_address': 'johndoe@exampledomain.com',
            'ip_address': '1.2.3.4'
        }

        #print(detector.predict('2021-11-13T12:18:21Z', event_variables)['ruleResults'])
        first_rule_result = detector.predict('2021-11-12T12:00:00Z', event_variables)['ruleResults'][0]['outcomes']
        # check list of outcomes is length gt 0
        assert len(first_rule_result) > 0
