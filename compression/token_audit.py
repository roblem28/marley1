import urllib.request, json
from abbrev import ConstructionCompressor

def tokencount(msg):
    payload = json.dumps({'content': msg}).encode()
    req = urllib.request.Request('http://100.97.87.86:8080/tokenize',
        data=payload, headers={'Content-Type':'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return len(resp['tokens'])

c = ConstructionCompressor()
terms = c._terms

better = []
neutral = []
worse = []

for full, abbr in terms.items():
    fc = tokencount(full)
    ac = tokencount(abbr)
    delta = ac - fc
    if delta < 0:
        better.append((full, abbr, fc, ac, delta))
    elif delta == 0:
        neutral.append((full, abbr, fc, ac, delta))
    else:
        worse.append((full, abbr, fc, ac, delta))

print(f'\n=== BETTER ({len(better)} terms) ===')
for full, abbr, fc, ac, d in sorted(better, key=lambda x: x[4]):
    print(f'  {full} ({fc}) -> {abbr} ({ac}) delta={d:+d}')

print(f'\n=== NEUTRAL ({len(neutral)} terms) ===')
for full, abbr, fc, ac, d in neutral:
    print(f'  {full} ({fc}) -> {abbr} ({ac}) delta={d:+d}')

print(f'\n=== WORSE ({len(worse)} terms) ===')
for full, abbr, fc, ac, d in sorted(worse, key=lambda x: x[4], reverse=True):
    print(f'  {full} ({fc}) -> {abbr} ({ac}) delta={d:+d}')

print(f'\nSUMMARY: {len(better)} better, {len(neutral)} neutral, {len(worse)} worse out of {len(terms)} terms')
