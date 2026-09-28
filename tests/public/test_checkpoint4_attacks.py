"""Check attack evidence generation without contacting a model."""
import asyncio
import json

from attacks import attacks


def test_attack_runner_preserves_success_and_api_error(tmp_path, monkeypatch):
    async def fake_chat(agent, runner, prompt):
        if prompt == 'unavailable':
            raise RuntimeError('provider unavailable')
        return 'db.vinbank.internal:5432', None

    monkeypatch.setattr(attacks, 'chat_with_agent', fake_chat)
    prompts = [
        {'id': 1, 'category': 'completion', 'input': 'complete'},
        {'id': 2, 'category': 'translation', 'input': 'unavailable'},
    ]
    rows = asyncio.run(attacks.run_attacks(None, None, prompts,
        target_name='red_default', output_path=tmp_path / 'detail.json'))
    assert rows[0]['leaked'] is True
    assert rows[1]['layer'] == 'error'
    assert rows[1]['blocked'] is False
    assert rows[1]['leaked'] is False
    detail = json.loads((tmp_path / 'detail.json').read_text())
    assert detail['errors'] == 1
    attacks.save_attack_results(unsafe_results=rows, filepath=tmp_path / 'summary.json')
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['summary']['unsafe_errors'] == 1
    assert summary['summary']['unsafe_leaked'] == 1
    assert summary['unsafe_attacks'][1]['error'] == 'RuntimeError: provider unavailable'
