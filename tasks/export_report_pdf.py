import os
import re
import subprocess


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_MD = os.path.join(ROOT_DIR, 'REPORT.md')
REPORT_TEX = os.path.join(ROOT_DIR, 'REPORT.tex')
REPORT_PDF = os.path.join(ROOT_DIR, 'REPORT.pdf')


def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def latex_escape(text):
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in text)


def apply_inline_formatting(text):
    placeholders = []

    def stash(pattern, formatter, source):
        def repl(match):
            placeholders.append(formatter(match.group(1)))
            return f'@@PLACEHOLDER{len(placeholders) - 1}@@'
        return re.sub(pattern, repl, source)

    text = stash(r'`([^`]+)`', lambda s: r'\texttt{' + latex_escape(s) + '}', text)
    text = stash(r'\*\*(.+?)\*\*', lambda s: r'\textbf{' + latex_escape(s) + '}', text)
    text = latex_escape(text)

    for idx, replacement in enumerate(placeholders):
        text = text.replace(latex_escape(f'@@PLACEHOLDER{idx}@@'), replacement)
    return text


def convert_table(table_lines):
    headers = [cell.strip() for cell in table_lines[0].strip('|').split('|')]
    alignments = [cell.strip() for cell in table_lines[1].strip('|').split('|')]
    rows = [[cell.strip() for cell in row.strip('|').split('|')] for row in table_lines[2:]]

    col_spec = []
    for align in alignments:
        if align.startswith(':') and align.endswith(':'):
            col_spec.append('c')
        elif align.endswith(':'):
            col_spec.append('r')
        else:
            col_spec.append('l')

    parts = []
    parts.append(r'\begin{longtable}{' + ''.join(col_spec) + '}')
    parts.append(r'\toprule')
    parts.append(' & '.join(apply_inline_formatting(cell) for cell in headers) + r' \\')
    parts.append(r'\midrule')
    parts.append(r'\endfirsthead')
    parts.append(r'\toprule')
    parts.append(' & '.join(apply_inline_formatting(cell) for cell in headers) + r' \\')
    parts.append(r'\midrule')
    parts.append(r'\endhead')
    for row in rows:
        parts.append(' & '.join(apply_inline_formatting(cell) for cell in row) + r' \\')
    parts.append(r'\bottomrule')
    parts.append(r'\end{longtable}')
    return '\n'.join(parts)


def convert_markdown(markdown):
    lines = markdown.splitlines()
    tex_lines = []
    in_itemize = False
    in_enumerate = False
    i = 0

    def close_lists():
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            tex_lines.append(r'\end{itemize}')
            in_itemize = False
        if in_enumerate:
            tex_lines.append(r'\end{enumerate}')
            in_enumerate = False

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            close_lists()
            tex_lines.append('')
            i += 1
            continue

        if stripped.startswith('# '):
            close_lists()
            i += 1
            continue

        if stripped.startswith('## '):
            close_lists()
            tex_lines.append(r'\section*{' + apply_inline_formatting(stripped[3:]) + '}')
            i += 1
            continue

        if stripped.startswith('### '):
            close_lists()
            tex_lines.append(r'\subsection*{' + apply_inline_formatting(stripped[4:]) + '}')
            i += 1
            continue

        image_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
        if image_match:
            close_lists()
            caption = apply_inline_formatting(image_match.group(1))
            image_path = image_match.group(2)
            tex_lines.append(r'\begin{figure}[htbp]')
            tex_lines.append(r'\centering')
            tex_lines.append(r'\includegraphics[width=0.95\linewidth]{' + latex_escape(image_path) + '}')
            tex_lines.append(r'\caption{' + caption + '}')
            tex_lines.append(r'\end{figure}')
            tex_lines.append('')
            i += 1
            continue

        if stripped.startswith('|') and stripped.endswith('|'):
            close_lists()
            table_lines = []
            while i < len(lines):
                candidate = lines[i].strip()
                if not (candidate.startswith('|') and candidate.endswith('|')):
                    break
                table_lines.append(candidate)
                i += 1
            tex_lines.append(convert_table(table_lines))
            tex_lines.append('')
            continue

        if stripped.startswith('- '):
            if in_enumerate:
                tex_lines.append(r'\end{enumerate}')
                in_enumerate = False
            if not in_itemize:
                tex_lines.append(r'\begin{itemize}')
                in_itemize = True
            tex_lines.append(r'\item ' + apply_inline_formatting(stripped[2:]))
            i += 1
            continue

        ordered_match = re.match(r'\d+\.\s+(.*)', stripped)
        if ordered_match:
            if in_itemize:
                tex_lines.append(r'\end{itemize}')
                in_itemize = False
            if not in_enumerate:
                tex_lines.append(r'\begin{enumerate}')
                in_enumerate = True
            tex_lines.append(r'\item ' + apply_inline_formatting(ordered_match.group(1)))
            i += 1
            continue

        close_lists()
        tex_lines.append(apply_inline_formatting(stripped) + r'\\')
        i += 1

    close_lists()
    return '\n'.join(tex_lines)


def build_document(title, body):
    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{parskip}}
\usepackage{{float}}
\setmainfont{{Noto Serif CJK SC}}
\setsansfont{{Noto Sans CJK SC}}
\setmonofont{{Noto Sans Mono CJK SC}}
\setCJKmainfont{{Noto Serif CJK SC}}
\hypersetup{{colorlinks=true, linkcolor=black, urlcolor=blue}}
\title{{{latex_escape(title)}}}
\date{{}}
\begin{{document}}
\maketitle
{body}
\end{{document}}
"""


def main():
    if not os.path.exists(REPORT_MD):
        raise FileNotFoundError(f'Missing report markdown: {REPORT_MD}')

    markdown = read_text(REPORT_MD)
    title_match = re.search(r'^#\s+(.+)$', markdown, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else 'Report'
    body = convert_markdown(markdown)
    latex = build_document(title, body)
    write_text(REPORT_TEX, latex)

    command = [
        'xelatex',
        '-interaction=nonstopmode',
        '-halt-on-error',
        '-output-directory',
        ROOT_DIR,
        REPORT_TEX,
    ]
    subprocess.run(command, check=True, cwd=ROOT_DIR)
    subprocess.run(command, check=True, cwd=ROOT_DIR)

    if not os.path.exists(REPORT_PDF):
        raise RuntimeError('PDF export did not produce REPORT.pdf')

    print(f'LaTeX written to {REPORT_TEX}')
    print(f'PDF written to {REPORT_PDF}')


if __name__ == '__main__':
    main()
