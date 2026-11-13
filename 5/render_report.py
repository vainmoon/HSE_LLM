import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import markdown
    from weasyprint import HTML

    body = markdown.markdown(args.source.read_text(), extensions=["tables", "fenced_code"])
    css = """
    @page { size: A4; margin: 16mm 13mm; @bottom-right { content: counter(page); font-size: 9pt; } }
    body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #202830; }
    h1 { font-size: 20pt; } h2 { font-size: 14pt; margin-top: 18pt; } h3 { font-size: 11pt; }
    h1, h2, h3 { break-after: avoid; } img { max-width: 100%; }
    table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 10pt 0; }
    th, td { border: 0.5pt solid #bdc6cd; padding: 4pt; text-align: left; }
    th { background: #edf2f7; } tr { break-inside: avoid; }
    pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.5pt; white-space: pre-wrap;
          overflow-wrap: anywhere; background: #f4f6f8; padding: 8pt; }
    code { font-family: 'DejaVu Sans Mono', monospace; font-size: 85%; overflow-wrap: anywhere; }
    a { color: #135e96; text-decoration: none; } blockquote { border-left: 3pt solid #889bac; padding-left: 10pt; }
    """
    html = f'<!doctype html><html lang="ru"><meta charset="utf-8"><style>{css}</style><body>{body}</body></html>'
    HTML(string=html, base_url=str(args.source.resolve().parent)).write_pdf(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
