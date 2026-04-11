#!/usr/bin/env python3
"""
Marley1 Chat Server — runs on Little Boy (Pi 5)
Serves marley_app.html, stores chats/agents/projects, proxies inference to Fat Man

Usage:
  python3 marley_server.py              # Tailscale-only (default)
  python3 marley_server.py --public     # bind 0.0.0.0 (LAN + Tailscale)
  python3 marley_server.py --port 9000  # custom port
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, uuid, time, urllib.request, urllib.error, sys, argparse

# ── Config ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--public', action='store_true')
parser.add_argument('--port', type=int, default=7842)
parser.add_argument('--fatman', default='http://100.97.87.86:8080/v1/chat/completions')
parser.add_argument('--cert', default=os.path.expanduser('~/marley1/cert.pem'))
parser.add_argument('--key',  default=os.path.expanduser('~/marley1/key.pem'))
args = parser.parse_args()

BIND      = '0.0.0.0' if args.public else '100.110.181.128'
PORT      = args.port
FAT_MAN   = args.fatman
STORE     = os.path.expanduser('~/marley1/chatstore')
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice', 'marley_app.html')
TAILSCALE_PREFIX = '100.'

for d in [STORE, f'{STORE}/chats', f'{STORE}/agents', f'{STORE}/projects']:
    os.makedirs(d, exist_ok=True)

# ── Storage helpers ──────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

def save(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=2)

def list_dir(path):
    if not os.path.exists(path): return []
    items = []
    for fn in sorted(os.listdir(path), reverse=True):
        if fn.endswith('.json'):
            d = load(f'{path}/{fn}')
            if d: items.append(d)
    return items

# ── Handler ──────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        ts = time.strftime('%H:%M:%S')
        print(f'[{ts}] {self.client_address[0]} {fmt % args}')

    def is_allowed(self):
        if args.public:
            return True
        ip = self.client_address[0]
        return ip.startswith(TAILSCALE_PREFIX) or ip == '127.0.0.1'

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path):
        if not os.path.exists(path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return
        with open(path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def body(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if not self.is_allowed():
            self.send_json(403, {'error': 'forbidden'})
            return
        p = self.path.split('?')[0]

        # Serve HTML
        if p in ('/', '/marley_app.html', '/index.html'):
            self.send_html(HTML_FILE)
            return

        # Status
        if p == '/api/status':
            self.send_json(200, {
                'status': 'ok',
                'mode': 'public' if args.public else 'tailscale',
                'fatman': FAT_MAN,
                'port': PORT,
                'uptime': round(time.time() - START_TIME)
            })
            return

        if p == '/api/chats':
            self.send_json(200, list_dir(f'{STORE}/chats'))
        elif p.startswith('/api/chats/'):
            d = load(f'{STORE}/chats/{p[11:]}.json')
            self.send_json(200 if d else 404, d or {})
        elif p == '/api/agents':
            self.send_json(200, list_dir(f'{STORE}/agents'))
        elif p.startswith('/api/agents/'):
            d = load(f'{STORE}/agents/{p[12:]}.json')
            self.send_json(200 if d else 404, d or {})
        elif p == '/api/projects':
            self.send_json(200, list_dir(f'{STORE}/projects'))
        elif p.startswith('/api/projects/'):
            d = load(f'{STORE}/projects/{p[14:]}.json')
            self.send_json(200 if d else 404, d or {})
        else:
            self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        if not self.is_allowed():
            self.send_json(403, {'error': 'forbidden'})
            return
        p = self.path
        b = self.body()

        if p == '/api/chats':
            cid = str(uuid.uuid4())[:8]
            chat = {
                'id': cid, 'title': b.get('title', 'New Chat'),
                'project_id': b.get('project_id'), 'agent_id': b.get('agent_id'),
                'created': time.time(), 'updated': time.time(), 'messages': []
            }
            save(f'{STORE}/chats/{cid}.json', chat)
            self.send_json(201, chat)

        elif p.startswith('/api/chats/') and p.endswith('/message'):
            cid = p[11:-8]
            chat = load(f'{STORE}/chats/{cid}.json')
            if not chat: return self.send_json(404, {})
            user_msg = {'role': 'user', 'content': b['content']}
            chat['messages'].append(user_msg)
            msgs = []
            if chat.get('agent_id'):
                agent = load(f'{STORE}/agents/{chat["agent_id"]}.json')
                if agent and agent.get('system_prompt'):
                    msgs.append({'role': 'system', 'content': agent['system_prompt']})
            msgs += chat['messages']
            try:
                req_body = json.dumps({
                    'model': 'qwen', 'messages': msgs, 'max_tokens': 500
                }).encode()
                req = urllib.request.Request(FAT_MAN, data=req_body,
                    headers={'Content-Type': 'application/json'})
                resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
                reply = resp['choices'][0]['message']['content']
            except Exception as e:
                reply = f'[Fat Man error: {e}]'
            assistant_msg = {'role': 'assistant', 'content': reply}
            chat['messages'].append(assistant_msg)
            chat['updated'] = time.time()
            if len(chat['messages']) == 2:
                chat['title'] = b['content'][:40] + ('...' if len(b['content']) > 40 else '')
            save(f'{STORE}/chats/{cid}.json', chat)
            self.send_json(200, {'reply': reply, 'chat': chat})

        elif p == '/api/agents':
            aid = str(uuid.uuid4())[:8]
            agent = {
                'id': aid, 'name': b.get('name', 'New Agent'),
                'system_prompt': b.get('system_prompt', ''),
                'description': b.get('description', ''), 'created': time.time()
            }
            save(f'{STORE}/agents/{aid}.json', agent)
            self.send_json(201, agent)

        elif p == '/api/projects':
            pid = str(uuid.uuid4())[:8]
            project = {
                'id': pid, 'name': b.get('name', 'New Project'),
                'description': b.get('description', ''), 'created': time.time()
            }
            save(f'{STORE}/projects/{pid}.json', project)
            self.send_json(201, project)

        else:
            self.send_json(404, {'error': 'not found'})

    def do_PUT(self):
        if not self.is_allowed():
            self.send_json(403, {'error': 'forbidden'})
            return
        p = self.path
        b = self.body()
        if p.startswith('/api/chats/'):
            cid = p[11:]
            chat = load(f'{STORE}/chats/{cid}.json')
            if not chat: return self.send_json(404, {})
            chat.update({k: v for k, v in b.items() if k in ('title','project_id','agent_id')})
            chat['updated'] = time.time()
            save(f'{STORE}/chats/{cid}.json', chat)
            self.send_json(200, chat)
        elif p.startswith('/api/agents/'):
            aid = p[12:]
            agent = load(f'{STORE}/agents/{aid}.json')
            if not agent: return self.send_json(404, {})
            agent.update({k: v for k, v in b.items() if k in ('name','system_prompt','description')})
            save(f'{STORE}/agents/{aid}.json', agent)
            self.send_json(200, agent)
        elif p.startswith('/api/projects/'):
            pid = p[14:]
            project = load(f'{STORE}/projects/{pid}.json')
            if not project: return self.send_json(404, {})
            project.update({k: v for k, v in b.items() if k in ('name','description')})
            save(f'{STORE}/projects/{pid}.json', project)
            self.send_json(200, project)
        else:
            self.send_json(404, {})

    def do_DELETE(self):
        if not self.is_allowed():
            self.send_json(403, {'error': 'forbidden'})
            return
        p = self.path
        if p.startswith('/api/chats/'):       path = f'{STORE}/chats/{p[11:]}.json'
        elif p.startswith('/api/agents/'):    path = f'{STORE}/agents/{p[12:]}.json'
        elif p.startswith('/api/projects/'): path = f'{STORE}/projects/{p[14:]}.json'
        else: return self.send_json(404, {})
        if os.path.exists(path):
            os.unlink(path)
            self.send_json(200, {'deleted': True})
        else:
            self.send_json(404, {})

# ── Start ─────────────────────────────────────────────────────────────────────
import ssl
START_TIME = time.time()
use_ssl = os.path.exists(args.cert) and os.path.exists(args.key)
scheme = 'https' if use_ssl else 'http'
mode = 'PUBLIC' if args.public else 'TAILSCALE-ONLY'
print(f'Marley1 server  port={PORT}  mode={mode}  ssl={use_ssl}')
print(f'Store:  {STORE}')
print(f'HTML:   {HTML_FILE}')
print(f'FatMan: {FAT_MAN}')
print(f'URL:    {scheme}://{BIND}:{PORT}/')

try:
    server = HTTPServer((BIND, PORT), Handler)
    if use_ssl:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        print('SSL enabled')
    server.serve_forever()
except KeyboardInterrupt:
    print('\nStopped.')
