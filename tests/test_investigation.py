"""Replay saved real reviews and verify the evidence bindings used by the planner."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from mechevidence import evaluate


class InvestigationFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = json.loads((Path(__file__).resolve().parents[1] /
                                 'examples/investigation_feedback.json').read_text())

    def test_real_inputs_replay_complete_results_and_request_references(self):
        snapshots = self.packet['snapshots']
        for case_id, snapshot in snapshots.items():
            with self.subTest(case_id=case_id):
                stages = []
                saved = snapshot['evaluation']

                def replay(stage, _instructions, _payload, _schema):
                    stages.append(stage)
                    return deepcopy(saved[stage])

                self.assertEqual(evaluate(snapshot['case'], replay), saved)
                self.assertEqual(stages, ['local_review', 'mechanism_review'])
                materials = {m['id']: m for m in snapshot['case']['materials']}
                for observation in saved['evidence']:
                    for ref in observation['references']:
                        material = materials[ref['material_id']]
                        self.assertEqual(material['text'][ref['start']:ref['end']], ref['text'])
                        self.assertEqual(material['version'], ref['version'])
                        selected = material.get('allowed_passage_ids')
                        if selected is not None:
                            self.assertIn(ref['id'], selected)
        for scenario in self.packet['scenarios']:
            request = scenario['evidence_request']
            result = snapshots[request['case_id']]['evaluation']
            claims = {row['id'] for row in result['claims']}
            joins = {row['id'] for row in result['joins']}
            observations = {row['id']: row for row in result['evidence']}
            self.assertLessEqual(set(request['claim_ids']), claims)
            self.assertLessEqual(set(request['join_ids']), joins)
            self.assertLessEqual(set(request['observation_ids']), observations.keys())
            self.assertEqual(request['scope'], result['scope'])
            gaps = [result['critical_gaps'][index] for index in request['gap_indices']]
            self.assertLessEqual(set(request['claim_ids']), {cid for gap in gaps for cid in gap['claim_ids']})
            self.assertLessEqual(set(request['join_ids']), {jid for gap in gaps for jid in gap['join_ids']})
            for anchor in request['starting_references']:
                refs = observations[anchor['observation_id']]['references']
                self.assertTrue(any(ref['id'] == anchor['passage_id'] and
                                    ref['material_id'] == anchor['material_id'] for ref in refs))

    def test_restoring_original_mediation_evidence_closes_only_the_stated_gap(self):
        before, after = [self.packet['snapshots'][key] for key in ('Q003', 'Q001')]
        for field in ('question', 'mechanism', 'scope', 'claims', 'joins'):
            self.assertEqual(before['case'][field], after['case'][field])
        left, right = before['case']['materials'][0], after['case']['materials'][0]
        self.assertEqual(left['text'], right['text'])
        self.assertEqual(left['version'], right['version'])
        self.assertNotIn('M1:P009', left['allowed_passage_ids'])
        self.assertIsNone(right.get('allowed_passage_ids'))
        self.assertEqual(before['evaluation']['joins'][0]['verdict'], 'unestablished')
        self.assertEqual(after['evaluation']['joins'][0]['verdict'], 'supported')
        self.assertEqual(before['evaluation']['overall']['verdict'], 'partial')
        self.assertEqual(after['evaluation']['overall']['verdict'], 'supported')
        self.assertFalse(after['evaluation']['critical_gaps'])
        for result in (before['evaluation'], after['evaluation']):
            self.assertEqual(result['independence']['distinct_source_groups'], 1)
            self.assertIsNone(result['independence']['independent_cohorts'])
        join = after['evaluation']['joins'][0]
        observations = {row['id']: row for row in after['evaluation']['evidence']}
        cited = {pid for oid in join['supporting_observation_ids'] for pid in observations[oid]['passage_ids']}
        self.assertIn('M1:P009', cited)

    def test_counterevidence_survives_when_there_is_no_paired_conflict(self):
        result = self.packet['snapshots']['Q020']['evaluation']
        claims = {row['id']: row for row in result['claims']}
        self.assertEqual(claims['C1']['verdict'], 'supported')
        self.assertEqual(claims['C2']['verdict'], 'refuted')
        self.assertEqual(claims['C3']['verdict'], 'insufficient')
        self.assertFalse(result['conflicts'])
        self.assertFalse(result['unresolved_counterevidence'])
        self.assertEqual(claims['C2']['opposing_observation_ids'], ['L2'])
        observations = {row['id']: row for row in result['evidence']}
        self.assertEqual(observations['L1']['passage_ids'], observations['L2']['passage_ids'])
        self.assertEqual(observations['L2']['relation'], 'opposes')
        self.assertEqual(result['overall']['verdict'], 'partial')
        self.assertTrue(all(row['verdict'] == 'unestablished' for row in result['joins']))

    def test_graph_assertions_cannot_replace_original_mechanism_evidence(self):
        snapshot = self.packet['snapshots']['Q001']
        case = deepcopy(snapshot['case'])
        for material in case['materials']:
            material['source_kind'] = 'graph'
        # Deliberately reuse a favourable semantic review; the program must still reject it.
        result = evaluate(case, lambda stage, *_: deepcopy(snapshot['evaluation'][stage]))
        self.assertEqual(result['overall']['verdict'], 'insufficient')
        self.assertTrue(all(not row['causal_evidence_eligible'] for row in result['evidence']))
        self.assertTrue(all(not row['supporting_observation_ids'] for row in result['claims'] + result['joins']))


if __name__ == '__main__':
    unittest.main()
