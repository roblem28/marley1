# Marley One GPT–Claude relay

The Pi takes Bob's message from a **private GitHub mailbox**, calls GPT, gives that
answer and the shared history to Claude, then returns Claude's contribution to
GPT for a final answer. Every hop is saved before the next starts. A request is
limited to three model calls. The existing Fat Man chat server is independent.

This connects the **OpenAI and Anthropic APIs**. It does not silently attach to an
existing ChatGPT/Claude browser conversation or inherit those apps' memories.
Send the relevant context into the mailbox. Only this mailbox's transcript is
shared with the two providers. API credentials and model access are required.

## Setup on the Pi

Python 3.9 or newer; no pip packages, inbound port, or GPU required.

After checking out the branch containing this directory in `~/marley1`:

```bash
cd ~/marley1
python3 relay/marley_relay.py configure --repo roblem28/marley-one --create-mailbox
python3 relay/marley_relay.py smoke
python3 relay/marley_relay.py chat
```

`configure` asks for the GitHub, OpenAI, and Anthropic credentials without echoing
them. Enter keys **on the Pi**, not in chat or a GitHub file. It verifies that the
mailbox is private, and stores settings in `~/.config/marley1/relay.json` with
permissions 0600. It creates the named private personal repo only when absent
and `--create-mailbox` was supplied. A public existing repo is rejected.

To use a repository-scoped GitHub token, create the private mailbox with an
initial README first, grant the token Contents read/write and Metadata read for
that repo, and omit `--create-mailbox`. Automatic creation needs a token that can
create personal private repositories. The source repo `roblem28/marley1` is
public and **must not be used as the conversation mailbox**.

Model IDs are configurable. The prompts offer documented examples
`gpt-6-astra` and `claude-sonnet-5-5`; use models available to your API accounts.
Each response has a default 2,048-output-token ceiling (including reasoning for
OpenAI). Set `max_output_tokens` in the local config, between 64 and 8,192, if a
model needs more. Incomplete responses halt the turn rather than masquerade as
a complete answer. `smoke` makes three live, potentially billable model calls
and succeeds only if all three return the same fresh, unpredictable nonce.

Environment variables override stored values: `GITHUB_TOKEN`,
`MARLEY_MAILBOX_REPO`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_API_KEY`, and
`ANTHROPIC_MODEL`. `doctor` verifies mailbox access and credential presence;
only `smoke` verifies live model connectivity.

## Send, receive, and continue

```bash
python3 relay/marley_relay.py roundtrip 'Review this plan and agree on a concrete next step.'
python3 relay/marley_relay.py send 'Continue with these corrections.' --conversation CONVERSATION_ID
python3 relay/marley_relay.py once CONVERSATION_ID
python3 relay/marley_relay.py history CONVERSATION_ID
```

`send` only queues; `once` performs the model calls. `roundtrip` does both.
`chat` keeps one shared conversation open until `/quit`. `-` in place of message
text reads stdin, avoiding shell escaping and shell history for longer inputs.
Record the printed conversation ID; queued work and partial results survive
terminal disconnects and Pi restarts.

An assistant with write access to the private mailbox can enqueue a version-1
file under `conversations/<32-lowercase-hex-id>.json`. Use a new UUID for a new
conversation. Example shape (replace the example ID):

```json
{
  "version": 1,
  "id": "0123456789abcdef0123456789abcdef",
  "created": "2026-11-14T00:00:00+00:00",
  "updated": "2026-11-14T00:00:00+00:00",
  "status": "queued",
  "route": ["gpt", "claude", "gpt"],
  "cursor": 0,
  "worker": null,
  "error": null,
  "messages": [{"speaker": "bob", "text": "Your request", "at": "2026-11-14T00:00:00+00:00"}]
}
```

For a follow-up, read the current file and blob SHA, require `status=complete`,
append Bob's new message, set `status=queued`, `cursor=0`, `worker=null`, and
`error=null`, then write using that SHA. Read the same file for progress and the
reply. Do not remove earlier messages or mark running work complete manually.

## Run continuously

```bash
mkdir -p ~/.config/systemd/user
cp relay/marley-relay.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now marley-relay
journalctl --user -u marley-relay -f
```

The supplied unit assumes the checkout is `~/marley1`; adjust its two paths for
another location. To keep a user service active after logout, enable lingering
for that user with `sudo loginctl enable-linger "$USER"`. Run the live smoke test
before starting the watcher. To stop: `systemctl --user stop marley-relay`.

## Recovery and concurrency

The current GitHub file SHA is the talking stick. A worker must atomically change
`queued` to `running` before any paid call. Other workers skip claimed turns. Each
saved response advances a cursor, so completed model turns are not replayed.
Provider failures become `failed`, with prior successful responses preserved.
The error log contains status, not provider response bodies or API keys.

```bash
python3 relay/marley_relay.py history CONVERSATION_ID
python3 relay/marley_relay.py retry CONVERSATION_ID
python3 relay/marley_relay.py once CONVERSATION_ID
```

A crash or interrupted write can leave `running`. Stop the previous worker,
inspect the saved history, then explicitly use `retry --abandon-running`.
There is no automatic lease expiry or automatic retry of billable calls. If a
provider answered but the process died before saving, that last call cannot be
proven delivered and may be billed again on explicit retry. Exactly-once API
execution across that failure window is not claimed.

History is retained without silent truncation. Above 200,000 transcript
characters, start a new conversation with an explicit summary. Archive completed
conversation files outside `conversations/` before reaching 1,000 files. GitHub
commit history also retains transcripts; this is a private mailbox, not an
ephemeral message store.

## Validation

```bash
python3 -m unittest discover -s relay -p 'test_*.py' -v
```

Tests exercise the route, shared history, provider HTTP contracts with local
test doubles, persistence/restart behavior, competing workers, failure recovery,
and private-mailbox enforcement. They do not prove live model or Pi access.

Provider contracts used:
- [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/python/messages/create)
- [Anthropic API versions](https://platform.claude.com/docs/en/api/versioning)
- [Claude model IDs](https://platform.claude.com/docs/en/models/overview)
