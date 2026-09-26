from pathlib import Path
import re
import pdfplumber

source = Path('papers/source')
target = Path('papers/parsed')
target.mkdir(parents=True, exist_ok=True)

skip = {'1704.04572v4.pdf'}

def clean(text: str) -> str:
    text = text.replace('\u00ad', '').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Rejoin words split at a line-ending hyphen while preserving ordinary hyphens.
    text = re.sub(r'(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

for pdf in sorted(source.glob('*.pdf')):
    if pdf.name in skip:
        continue
    with pdfplumber.open(pdf) as doc:
        pages = [clean(page.extract_text(x_tolerance=2, y_tolerance=3) or '') for page in doc.pages]
    nonempty = [p for p in pages if p]
    first_lines = nonempty[0].splitlines() if nonempty else [pdf.stem]
    title = first_lines[0].strip() if first_lines else pdf.stem
    # Some papers place a page number before the title.
    if re.fullmatch(r'\d+', title) and len(first_lines) > 1:
        title = first_lines[1].strip()
    body = [f'# {title}', '', f'> Source: `{pdf.name}`', '',
            '> This Markdown was generated from the PDF text layer. Page boundaries are retained for traceability.', '']
    for i, page in enumerate(pages, 1):
        body.extend([f'<!-- Page {i} -->', '', page, ''])
    out = target / f'{pdf.stem}.md'
    out.write_text('\n'.join(body).rstrip() + '\n', encoding='utf-8')
    print(out, len(pages), sum(len(p) for p in pages))
