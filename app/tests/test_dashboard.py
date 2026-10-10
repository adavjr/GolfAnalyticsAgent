from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from golf_agent.agent import ToolRuntime
from golf_agent.store import LocalStore, AstraStore

ROOT = Path(__file__).resolve().parents[2]

class DashboardTests(unittest.TestCase):
    def test_scope_blocks_other_player_before_storage(self):
        class Store:
            def history(self, *args):
                raise AssertionError('Must not reach storage')
        runtime = ToolRuntime(Store(), None, ROOT, selected_player='Player A')
        with self.assertRaisesRegex(ValueError, 'restricted'):
            runtime.dispatch('query_player_history', {'player_id': 'Player B', 'limit': 30})

    def test_roster(self):
        store = LocalStore(ROOT / 'data/trackman_mock.csv')
        self.assertEqual(store.list_players(), ['Player A', 'Player B', 'Player C'])

    def test_unrelated_query_shows_error_without_agent(self):
        with patch.object(AstraStore, '__init__', return_value=None), \
             patch.object(AstraStore, 'list_players', return_value=['Player A']), \
             patch('golf_agent.agent.run_agent') as agent:
            at = AppTest.from_file(str(ROOT / 'app/dashboard.py')).run(timeout=20)
            at.text_area[0].input('Write a pasta recipe')
            next(b for b in at.button if b.label == 'Analyze player').click().run(timeout=20)
            self.assertFalse(at.exception)
            agent.assert_not_called()
            self.assertIn('does not relate to the available golf performance data', at.error[0].value)
            self.assertFalse(next(b for b in at.button if b.label == 'Analyze player').disabled)

    def test_busy_state_resets_after_failure(self):
        import streamlit as st
        def fail(*args, **kwargs):
            self.assertTrue(st.session_state.running)
            raise RuntimeError("Simulated API failure")
        with patch.object(AstraStore, '__init__', return_value=None), \
             patch.object(AstraStore, 'list_players', return_value=['Player A']), \
             patch('golf_agent.agent.run_agent', side_effect=fail) as agent:
            at = AppTest.from_file(str(ROOT / 'app/dashboard.py')).run(timeout=20)
            at.text_area[0].input('Analyze my delivery')
            next(b for b in at.button if b.label == 'Analyze player').click().run(timeout=20)
            self.assertFalse(at.exception)
            self.assertEqual(agent.call_count, 1)
            self.assertFalse(at.session_state['running'])
            self.assertFalse(next(b for b in at.button if b.label == 'Analyze player').disabled)
            self.assertEqual(len(at.error), 1)

    def test_ui_submit_and_switch(self):
        brief = {'player_id': 'Player A', 'summary': 'Demo evidence', 'swing_tendency': 'Neutral',
                 'evidence': ['30 shots'], 'equipment_assessment': {'reason': 'No trials',
                 'suggested_fitting_trial': 'Compare shafts'}, 'charts': [], 'limitations': [], 'next_steps': []}
        with patch.object(AstraStore, '__init__', return_value=None), \
             patch.object(AstraStore, 'list_players', return_value=['Player A', 'Player B']), \
             patch('golf_agent.agent.run_agent', return_value=brief) as agent:
            at = AppTest.from_file(str(ROOT / 'app/dashboard.py')).run(timeout=20)
            self.assertFalse(at.exception)
            at.text_area[0].input('Analyze my delivery')
            next(b for b in at.button if b.label == 'Analyze player').click().run(timeout=20)
            self.assertFalse(at.exception)
            self.assertEqual(agent.call_count, 1)
            self.assertEqual(agent.call_args.args[1].selected_player, 'Player A')
            self.assertTrue(any(m.value == 'Demo evidence' for m in at.markdown))
            at.selectbox(key='player').select('Player B').run()
            self.assertFalse(at.exception)
            self.assertEqual(at.text_area[0].value, '')
            self.assertFalse(any(m.value == 'Demo evidence' for m in at.markdown))
            self.assertEqual(agent.call_count, 1)

if __name__ == '__main__':
    unittest.main()
