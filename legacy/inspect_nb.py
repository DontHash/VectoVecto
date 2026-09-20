
# repo root: legacy/ scripts import modules that live at the repo root
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import json

nb = json.load(open('kaggle_pulled/vectovecto-tier-b-drunet-training.ipynb'))
print(f'cells: {len(nb["cells"])}')
print()
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code':
        print(f'=== CELL {i} (code) ===')
        for line in src.split('\n')[:2]:
            print('  CODE:', line[:100])
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
                print('  STDOUT (last 1000):')
                for line in text[-1000:].split('\n'):
                    print('    ', line)
            elif out.get('output_type') == 'error':
                print('  ERROR:', out.get('ename', '?'), '-', out.get('evalue', '?'))
                tb = out.get('traceback', [])
                if tb:
                    last = '\n'.join(tb[-5:])
                    print('  TRACEBACK (last):')
                    for line in last[-1500:].split('\n'):
                        print('    ', line)
        print()
    elif cell['cell_type'] == 'markdown':
        first = src.split('\n')[0]
        print(f'=== CELL {i} (markdown): {first[:80]} ===')
        print()