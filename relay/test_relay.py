import base64
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import marley_relay as relay


CONFIG = {'repo': 'example/private-mailbox', 'github_token': 'fake-github-key',
          'openai_key': 'fake-openai-key', 'anthropic_key': 'fake-anthropic-key',
          'openai_model': 'test-gpt', 'anthropic_model': 'test-claude'}


class MemoryMailbox:
    def __init__(self):
        self.docs = {}
        self.version = 0
        self.conflict_next = False
        self.reject_cursor = None

    def check_private(self):
        pass

    def get(self, cid):
        return copy.deepcopy(self.docs[cid])

    def put(self, doc, sha=None):
        relay.validate(doc, doc['id'])
        if self.conflict_next:
            self.conflict_next = False
            raise relay.Conflict('Another worker won')
        if self.reject_cursor is not None and doc['cursor'] == self.reject_cursor:
            raise relay.RelayError('Write unavailable')
        old_sha = self.docs.get(doc['id'], ({}, None))[1]
        if sha != old_sha:
            raise relay.Conflict('stale SHA')
        self.version += 1
        sha = str(self.version)
        self.docs[doc['id']] = copy.deepcopy(doc), sha
        return sha

    def ids(self):
        return list(self.docs)


class FakeProviders:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def ready(self, route):
        pass

    def reply(self, speaker, messages, final):
        self.calls.append((speaker, copy.deepcopy(messages), final))
        if len(self.calls) == self.fail_at:
            raise relay.RelayError('Provider unavailable')
        return {'speaker': speaker, 'text': 'Answer ' + str(len(self.calls)), 'at': relay.now()}


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.mailbox = MemoryMailbox()
        self.cid = relay.enqueue(self.mailbox, 'Keep the number 143 and the name Marley One.')

    def test_full_roundtrip_shared_history_and_no_automatic_loop(self):
        providers = FakeProviders()
        self.assertTrue(relay.run_one(self.mailbox, providers, self.cid))
        self.assertEqual([p[0] for p in providers.calls], ['gpt', 'claude', 'gpt'])
        self.assertEqual([len(p[1]) for p in providers.calls], [1, 2, 3])
        self.assertEqual([p[2] for p in providers.calls], [False, False, True])
        self.assertEqual(providers.calls[2][1][2]['speaker'], 'claude')
        self.assertFalse(relay.run_one(self.mailbox, providers, self.cid))
        self.assertEqual(len(providers.calls), 3)
        doc, _ = self.mailbox.get(self.cid)
        self.assertEqual(doc['status'], 'complete')
        self.assertEqual(doc['cursor'], 3)

    def test_followup_keeps_both_models_history(self):
        relay.run_one(self.mailbox, FakeProviders(), self.cid)
        cid = relay.enqueue(self.mailbox, 'Now check the schedule.', self.cid)
        providers = FakeProviders()
        relay.run_one(self.mailbox, providers, cid)
        first_context = providers.calls[0][1]
        self.assertEqual([m['speaker'] for m in first_context], ['bob', 'gpt', 'claude', 'gpt', 'bob'])
        self.assertIn('143', first_context[0]['text'])
        self.assertEqual(len(self.mailbox.get(cid)[0]['messages']), 8)

    def test_two_workers_only_winner_makes_calls(self):
        providers = FakeProviders()
        self.mailbox.conflict_next = True
        with self.assertRaises(relay.Conflict):
            relay.run_one(self.mailbox, providers, self.cid)
        self.assertEqual(providers.calls, [])

    def test_running_turn_cannot_be_taken_or_appended(self):
        doc, sha = self.mailbox.get(self.cid)
        doc['status'] = 'running'
        self.mailbox.put(doc, sha)
        self.assertFalse(relay.run_one(self.mailbox, FakeProviders(), self.cid))
        with self.assertRaises(relay.RelayError):
            relay.enqueue(self.mailbox, 'Another request', self.cid)
        with self.assertRaises(relay.RelayError):
            relay.retry(self.mailbox, self.cid)

    def test_failure_retains_reply_and_retry_resumes_at_claude(self):
        with self.assertRaises(relay.RelayError):
            relay.run_one(self.mailbox, FakeProviders(fail_at=2), self.cid)
        failed, _ = self.mailbox.get(self.cid)
        self.assertEqual((failed['status'], failed['cursor'], len(failed['messages'])), ('failed', 1, 2))
        relay.retry(self.mailbox, self.cid)
        providers = FakeProviders()
        relay.run_one(self.mailbox, providers, self.cid)
        self.assertEqual([p[0] for p in providers.calls], ['claude', 'gpt'])
        self.assertEqual(len(self.mailbox.get(self.cid)[0]['messages']), 4)

    def test_lost_write_after_paid_call_keeps_claim_locked(self):
        self.mailbox.reject_cursor = 1
        providers = FakeProviders()
        with self.assertRaises(relay.RelayError):
            relay.run_one(self.mailbox, providers, self.cid)
        doc, _ = self.mailbox.get(self.cid)
        self.assertEqual((doc['status'], doc['cursor']), ('running', 0))
        self.assertFalse(relay.run_one(self.mailbox, providers, self.cid))
        self.assertEqual(len(providers.calls), 1)
        self.mailbox.reject_cursor = None
        relay.retry(self.mailbox, self.cid, abandon_running=True)
        self.assertEqual(self.mailbox.get(self.cid)[0]['status'], 'queued')

    def test_missing_provider_credentials_does_not_claim_work(self):
        with self.assertRaises(relay.RelayError):
            relay.run_one(self.mailbox, relay.Providers({}), self.cid)
        self.assertEqual(self.mailbox.get(self.cid)[0]['status'], 'queued')

    def test_invalid_routes_and_paths_are_rejected(self):
        doc, _ = self.mailbox.get(self.cid)
        for route in (['gpt'] * 4, ['unknown'], [], 'gpt'):
            doc['route'] = route
            with self.assertRaises(relay.RelayError):
                relay.validate(doc, self.cid)
        for cid in ('../config', 'a' * 31, 'z' * 32):
            with self.assertRaises(relay.RelayError):
                relay.safe_id(cid)

    def test_overlarge_history_is_not_silently_truncated(self):
        providers = relay.Providers(CONFIG, transport=lambda *a, **k: self.fail('Must not call provider'))
        with self.assertRaisesRegex(relay.RelayError, 'Nothing was truncated'):
            providers.reply('gpt', [{'speaker': 'bob', 'text': 'x' * 200_001}], False)

    def test_unsafe_config_permissions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'relay.json'
            path.write_text(json.dumps(CONFIG))
            path.chmod(0o644)
            with self.assertRaises(relay.RelayError):
                relay.read_config(path)
            path.chmod(0o600)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'override-key'}):
                self.assertEqual(relay.read_config(path)['openai_key'], 'override-key')

    def test_incomplete_and_empty_provider_responses_stop(self):
        cases = [
            ('gpt', {'status': 'incomplete', 'output': []}),
            ('gpt', {'status': 'completed', 'output': []}),
            ('claude', {'stop_reason': 'max_tokens', 'content': [{'type': 'text', 'text': 'partial'}]}),
            ('claude', {'stop_reason': 'end_turn', 'content': []}),
        ]
        for speaker, response in cases:
            providers = relay.Providers(CONFIG, transport=lambda *a, **k: response)
            with self.assertRaises(relay.RelayError):
                providers.reply(speaker, [{'speaker': 'bob', 'text': 'hi'}], True)

    def test_smoke_command_checks_every_nonce_hop(self):
        class Echo(FakeProviders):
            def reply(self, speaker, messages, final):
                return {'speaker': speaker, 'text': messages[0]['text'].split(': ')[-1], 'at': relay.now()}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(relay, 'GitHubMailbox', return_value=self.mailbox), \
                patch.object(relay, 'Providers', return_value=Echo()), \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()):
            code = relay.main(['--config', str(Path(folder) / 'absent'), 'smoke'])
        self.assertEqual(code, 0)
        self.assertIn('LIVE ROUND TRIP VERIFIED', stdout.getvalue())
        # The test injected fake providers; this assertion verifies the CLI gate only.


class HTTPIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.files = {}
        self.private = True
        self.counter = 0
        self.api_status = None
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, value):
                data = json.dumps(value).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == '/github/repos/example/private-mailbox':
                    return self.send(200, {'private': case.private, 'default_branch': 'main'})
                if path.endswith('/contents/conversations'):
                    return self.send(200, [{'type': 'file', 'name': k + '.json'} for k in case.files])
                cid = Path(path).stem
                if cid not in case.files:
                    return self.send(404, {})
                return self.send(200, case.files[cid])

            def do_PUT(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cid = Path(urlsplit(self.path).path).stem
                old = case.files.get(cid)
                if (old and old['sha'] != body.get('sha')) or (not old and body.get('sha')):
                    return self.send(409, {})
                case.counter += 1
                item = {'sha': 'sha' + str(case.counter), 'content': body['content']}
                case.files[cid] = item
                return self.send(200, {'content': {'sha': item['sha']}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                case.records.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if case.api_status:
                    return self.send(case.api_status, {'error': 'NEVER_ECHO_SECRET_PROVIDER_BODY'})
                if self.path == '/openai/v1/responses':
                    transcript = json.loads(body['input'])
                    text = 'GPT read ' + str(len(transcript)) + ' messages'
                    return self.send(200, {'id': 'gpt-response', 'status': 'completed', 'model': 'test-gpt',
                        'output': [{'type': 'reasoning'}, {'type': 'message', 'content': [
                            {'type': 'output_text', 'text': text}]}], 'usage': {'output_tokens': 8}})
                transcript = json.loads(body['messages'][0]['content'])
                return self.send(200, {'id': 'claude-response', 'model': 'test-claude', 'stop_reason': 'end_turn',
                    'content': [{'type': 'thinking', 'thinking': 'not saved'},
                                {'type': 'text', 'text': 'Claude saw ' + transcript[-1]['text']}],
                    'usage': {'output_tokens': 10}})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def transport(self, method, url, headers, body=None, service='API'):
        parsed = urlsplit(url)
        prefix = {'api.github.com': '/github', 'api.openai.com': '/openai', 'api.anthropic.com': '/anthropic'}[parsed.netloc]
        local = 'http://127.0.0.1:' + str(self.server.server_port) + prefix + parsed.path
        if parsed.query:
            local += '?' + parsed.query
        return relay.http_json(method, local, headers, body, service)

    def test_http_roundtrip_survives_new_worker_and_obeys_provider_contracts(self):
        mailbox = relay.GitHubMailbox(CONFIG, self.transport)
        cid = relay.enqueue(mailbox, 'Check 143 MW. Preserve Marley One.')
        self.assertEqual(mailbox.ids(), [cid])
        # Fresh objects simulate a separate Pi worker, loading solely from remote state.
        worker = relay.GitHubMailbox(CONFIG, self.transport)
        relay.run_one(worker, relay.Providers(CONFIG, self.transport), cid)
        doc, _ = relay.GitHubMailbox(CONFIG, self.transport).get(cid)
        self.assertEqual(doc['status'], 'complete')
        self.assertEqual(doc['messages'][-1]['text'], 'GPT read 3 messages')
        self.assertEqual(doc['messages'][2]['text'], 'Claude saw GPT read 1 messages')
        self.assertEqual(len(self.records), 3)
        _, gpt_headers, gpt_body = self.records[0]
        _, claude_headers, claude_body = self.records[1]
        self.assertEqual(gpt_headers['authorization'], 'Bearer fake-openai-key')
        self.assertFalse(gpt_body['store'])
        self.assertEqual(gpt_body['max_output_tokens'], 2048)
        self.assertEqual(claude_headers['anthropic-version'], '2026-10-21')
        self.assertEqual(claude_headers['x-api-key'], 'fake-anthropic-key')
        self.assertEqual(claude_body['max_tokens'], 2048)
        self.assertNotIn('fake-openai-key', json.dumps(doc))
        self.assertNotIn('not saved', json.dumps(doc))

    def test_stale_github_sha_cannot_overwrite_a_claim(self):
        mailbox = relay.GitHubMailbox(CONFIG, self.transport)
        cid = relay.enqueue(mailbox, 'hi')
        first, sha = mailbox.get(cid)
        stale = copy.deepcopy(first)
        first['status'] = 'running'
        mailbox.put(first, sha)
        with self.assertRaises(relay.Conflict):
            mailbox.put(stale, sha)

    def test_public_mailbox_rejected_before_reading_or_writing_conversations(self):
        self.private = False
        with self.assertRaisesRegex(relay.RelayError, 'PRIVATE'):
            relay.GitHubMailbox(CONFIG, self.transport)
        self.assertEqual(self.files, {})

    def test_visibility_change_blocks_next_write(self):
        mailbox = relay.GitHubMailbox(CONFIG, self.transport)
        cid = relay.enqueue(mailbox, 'hi')
        doc, sha = mailbox.get(cid)
        self.private = False
        doc['status'] = 'running'
        with self.assertRaisesRegex(relay.RelayError, 'PRIVATE'):
            mailbox.put(doc, sha)

    def test_provider_errors_are_not_retried_and_bodies_are_not_exposed(self):
        self.api_status = 429
        provider = relay.Providers(CONFIG, self.transport)
        with self.assertRaises(relay.HTTPFailure) as failure:
            provider.reply('gpt', [{'speaker': 'bob', 'text': 'hi'}], True)
        self.assertEqual(len(self.records), 1)
        self.assertNotIn('NEVER_ECHO', str(failure.exception))
        self.assertNotIn('fake-openai', str(failure.exception))


if __name__ == '__main__':
    unittest.main()
