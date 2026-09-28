"""Offline checks for rate windows, observability and CP3 integration."""
import asyncio
import json
from types import SimpleNamespace

import jsonschema
import pytest

from assignment import pipeline
from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.rate_limiter import RateLimitPlugin
from core.openai_runtime import OpenAIRunner


def test_rate_window_and_user_isolation(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('assignment.rate_limiter.time.monotonic', lambda: now[0])
    limiter = RateLimitPlugin(max_requests=2, window_seconds=60)

    async def send(user):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user), user_message=None)

    async def run():
        assert await send('a') is None
        now[0] = 110
        assert await send('a') is None
        assert await send('a') is not None
        assert await send('b') is None
        now[0] = 160
        assert await send('a') is None
        assert await send('a') is not None
        assert list(limiter.user_windows['a']) == [110, 160]

    asyncio.run(run())
    assert (limiter.total_count, limiter.blocked_count) == (6, 2)


def test_audit_correlation_and_export(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr('assignment.audit_log.time.monotonic', lambda: now[0])
    audit = AuditLogPlugin()
    for request in ['a', 'b']:
        audit.record_input(user_id='user', text=request, request_id=request)
    now[0] = 10.25
    audit.record_output(user_id='user', text='blocked', request_id='b', blocked=True, layer='input_guardrail')
    audit.record_output(user_id='user', text='ok', request_id='a')
    assert [r['input'] for r in audit.logs] == ['b', 'a']
    assert audit.logs[0]['latency_ms'] == 250
    assert not audit._open
    path = tmp_path / 'nested' / 'audit.json'
    audit.export_json(str(path))
    assert json.loads(path.read_text()) == audit.logs


def test_monitoring_thresholds_and_no_duplicate_alerts(tmp_path):
    monitor = MonitoringAlert()
    assert monitor.check_metrics() == []
    monitor.total_requests = 10
    monitor.blocked_requests = 6
    monitor.rate_limit_hits = 6
    monitor.judge_checks = 10
    monitor.judge_fails = 4
    assert len(monitor.check_metrics()) == 3
    assert len(monitor.check_metrics()) == 3
    path = tmp_path / 'metrics.json'
    monitor.export_json(str(path))
    assert json.loads(path.read_text())['block_rate'] == .6


@pytest.mark.parametrize('destination', [
    'http://api.vinbank.example/x', 'https://api.vinbank.example.evil.com/x',
    'https://evil.example/x', 'https://user@api.vinbank.example/x',
    'https://api.vinbank.example:444/x', 'https://api.vinbank.example:bad/x',
    'https://api.vinbank.example\n.evil.com/x', '//api.vinbank.example/x',
])
def test_egress_rejects_untrusted_urls(destination):
    assert not pipeline.is_egress_allowed(destination, 'transfer amount 500000')


@pytest.mark.parametrize('payload', [
    'password is example', 'sk-test-key', 'db.vinbank.internal:5432',
    '0901234567', 'name@example.com', 'admin123', 'sk-\u200btest-key',
    'api_key: example',
])
def test_egress_sensitive_payload(payload):
    assert not pipeline.is_egress_allowed('https://api.vinbank.example/x', payload)


def test_egress_approved_payload():
    for host in ['api.vinbank.example', 'cases.vinbank.example']:
        assert pipeline.is_egress_allowed(f'https://{host}:443/x', 'approved transfer amount 500000')


def test_suite_uses_plugins_and_exports_consistent_results(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'OUTPUT_DIR', tmp_path)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='VinBank can help you with this banking request.'
        ))])

    monkeypatch.setattr(OpenAIRunner, '_client', lambda self: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    plugins = pipeline.build_production_plugins()
    audit, monitor = pipeline.build_observability()
    result = asyncio.run(pipeline.run_assignment_suite({'plugins': plugins, 'audit': audit, 'monitor': monitor}))
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[2] / 'schemas/results.schema.json').read_text())
    jsonschema.validate(result, schema)
    assert not any(q['blocked'] for q in result['safe_queries'])
    assert all(q['blocked'] and q['layer'] == 'input_guardrail' for q in result['attack_queries'])
    assert result['rate_limit']['passed'] == 10
    assert result['rate_limit']['blocked'] == 2
    assert len(calls) == 7  # five safe, two benign edge cases; burst never calls API
    assert len(audit.logs) == monitor.total_requests == 28
    assert monitor.blocked_requests == sum(r['blocked'] for r in audit.logs)
    assert monitor.rate_limit_hits == 2
    assert set(p.name for p in tmp_path.iterdir()) == {'results.json', 'audit_log.json', 'metrics.json'}


def test_model_failure_is_audited_without_success_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'OUTPUT_DIR', tmp_path)

    async def fail(self, *args, **kwargs):
        raise RuntimeError('provider unavailable')

    monkeypatch.setattr(OpenAIRunner, 'chat', fail)
    audit, monitor = pipeline.build_observability()
    with pytest.raises(RuntimeError, match='provider unavailable'):
        asyncio.run(pipeline.run_assignment_suite({
            'plugins': pipeline.build_production_plugins(), 'audit': audit, 'monitor': monitor,
        }))
    assert not (tmp_path / 'results.json').exists()
    assert audit.logs[0]['layer'] == 'runtime_error'
    assert not audit._open
    assert (tmp_path / 'audit_log.json').exists()
    assert (tmp_path / 'metrics.json').exists()
