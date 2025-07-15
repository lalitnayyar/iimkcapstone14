import markdown2
from fpdf import FPDF
from bs4 import BeautifulSoup

# Ensure DejaVuSans.ttf is in the same directory as this script.
FONT_PATH = 'ttf/DejaVuSans.ttf'

# Read markdown file
with open('lalitnayyar_capstone14seca.md', encoding='utf-8') as f:
    md = f.read()

# Convert markdown to HTML
html = markdown2.markdown(md)
soup = BeautifulSoup(html, 'html.parser')

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_font('DejaVu', '', 'ttf/DejaVuSans.ttf')
pdf.add_font('DejaVu', 'B', 'ttf/DejaVuSans-Bold.ttf')
pdf.set_font('DejaVu', '', 12)

for p in soup.find_all(['h1','h2','h3','h4','p','li','pre','code']):
    txt = p.get_text()
    if p.name == 'h1':
        pdf.set_font('DejaVu', 'B', 18)
    elif p.name == 'h2':
        pdf.set_font('DejaVu', 'B', 16)
    elif p.name == 'h3':
        pdf.set_font('DejaVu', 'B', 14)
    elif p.name in ['pre', 'code']:
        pdf.set_font('DejaVu', '', 9)
        # For code: wrap long lines
        for line in txt.splitlines():
            # Skip pathological code lines
            if any(len(word) > 40 for word in line.split()):
                print(f"[WARNING] Skipping pathological code line: {line[:60]}...")
                continue
            while len(line) > 100:
                try:
                    pdf.multi_cell(0, 5, line[:100])
                except Exception as e:
                    print(f"[WARNING] Skipping code chunk due to PDF error: {repr(e)} | {line[:60]}...")
                    break
                line = line[100:]
            if line.strip():
                try:
                    pdf.multi_cell(0, 5, line)
                except Exception as e:
                    print(f"[WARNING] Skipping code line due to PDF error: {repr(e)} | {line[:60]}...")
                    continue
        pdf.ln(1)
        continue
    else:
        pdf.set_font('DejaVu', '', 12)
    for line in txt.splitlines():
        if not line.strip():
            continue
        # Split line into words and check for very long words
        words = line.split()
        safe_line = ''
        for word in words:
            if len(word) > 80:
                # Truncate or split long words to avoid fpdf2 errors
                while len(word) > 80:
                    pdf.multi_cell(0, 10, word[:80])
                    word = word[80:]
                safe_line += word + ' '
            else:
                safe_line += word + ' '
        safe_line = safe_line.rstrip()
        if safe_line:
            # If the line is a single word longer than 40 chars, skip and warn
            if any(len(word) > 40 for word in safe_line.split()):
                print(f"[WARNING] Skipping pathological line (unbreakable word): {safe_line[:60]}...")
                continue
            try:
                pdf.multi_cell(0, 10, safe_line)
            except Exception as e:
                print(f"[WARNING] Skipping normal text line due to PDF rendering error: {repr(e)} | {safe_line[:60]}...")
                continue
    pdf.ln(1)

pdf.output('lalitnayyar_capstone14seca.pdf')
print('PDF created: lalitnayyar_capstone14seca.pdf')
