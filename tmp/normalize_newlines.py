from pathlib import Path
import re

root = Path('papers/parsed')

def normalize_page(page: str) -> str:
    lines = [line.rstrip() for line in page.splitlines()]
    out = []
    for line in lines:
        if not line.strip():
            if out and out[-1] != '':
                out.append('')
            continue
        line = line.strip()
        if out and out[-1] != '':
            prev = out[-1]
            # PDF line wrapping: a lowercase/digit continuation belongs to the
            # preceding line. Keep headings, lists, equations, and table rows.
            continuation = (
                re.match(r'^[a-z0-9(\["“]', line) is not None
                and not re.match(r'^[-*+] ', prev)
                and not prev.startswith('|')
                and not prev.endswith(('$$', '\\', ':'))
            )
            if continuation:
                if prev.endswith('-') and not prev.endswith(' --'):
                    out[-1] = prev[:-1] + line
                else:
                    out[-1] = prev + ' ' + line
                continue
        out.append(line)
    # Remove duplicate blank lines introduced at page boundaries.
    return '\n'.join(out).strip()

for path in sorted(root.glob('*.md')):
    text = path.read_text(encoding='utf-8')
    chunks = re.split(r'(<!-- Page \d+ -->\n)', text)
    for i in range(2, len(chunks), 2):
        chunks[i] = normalize_page(chunks[i])
    result = ''.join(chunks)
    result = re.sub(r'\s*<!-- Page (\d+) -->\s*', r'\n\n<!-- Page \1 -->\n\n', result)
    path.write_text(result.rstrip() + '\n', encoding='utf-8')
    print(path)
