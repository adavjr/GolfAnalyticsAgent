import unittest
from unittest.mock import Mock
from golf_agent.agent import run_agent
from golf_agent.relevance import validate_relevance, UnrelatedQueryError

class RelevanceTests(unittest.TestCase):
    def test_supported_queries(self):
        for prompt in ['Analyze Player A', 'How consistent are they?', 'Show a chart',
                       'Would a stiffer shaft reduce spin?', 'Summarize recent results']:
            validate_relevance(prompt)

    def test_unrelated_queries_never_call_api(self):
        client = Mock()
        for prompt in ['Write a pasta recipe', 'What is the capital of France?',
                       'Tell me a joke', 'Explain quantum physics']:
            with self.assertRaises(UnrelatedQueryError):
                run_agent(prompt, Mock(), 'test-model', client=client)
        client.responses.create.assert_not_called()
