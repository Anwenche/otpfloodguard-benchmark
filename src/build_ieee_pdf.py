#!/usr/bin/env python3
"""Build an IEEE/PRAI-style LaTeX PDF from the current manuscript markdown."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript.md"
FIGURES = ROOT / "figures"
OUT = ROOT / "build" / "ieee_submission"
TEMPLATE = ROOT / "build" / "ieee_template" / "conference-latex-template_10-17-19" / "IEEEtran.cls"
OUTPUT_PDF_NAME = "OTPFloodGuard_PRAI2026_Manuscript.pdf"


def latex_escape(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    text = text.replace("+/-", r"$\pm$")
    text = text.replace("->", r"$\rightarrow$")
    return text


def inline_format(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + latex_escape(part[1:-1]) + "}")
        else:
            protected: list[tuple[str, str]] = []

            def protect(match: re.Match[str]) -> str:
                token = f"@@RAWLATEX{len(protected)}@@"
                protected.append((token, match.group(0)))
                return token

            raw_safe = re.sub(r"\\footnote\{\\url\{[^{}]+\}\}", protect, part.replace("**", ""))
            raw_safe = re.sub(r"\\footnote\{\\href\{[^{}]+\}\{[^{}]+\}\}", protect, raw_safe)
            raw_safe = re.sub(r"\\url\{[^{}]+\}", protect, raw_safe)
            raw_safe = re.sub(r"\\\([^$]*?\\\)", protect, raw_safe)
            escaped = latex_escape(raw_safe)
            for token, raw in protected:
                escaped = escaped.replace(latex_escape(token), raw)
            out.append(escaped)
    return "".join(out)


def clean_heading(text: str) -> str:
    text = re.sub(r"^[IVX]+\.\s+", "", text)
    text = re.sub(r"^[A-Z]\.\s+", "", text)
    return text


def strip_fig_prefix(caption: str) -> str:
    return re.sub(r"^Fig\.\s*\d+\.\s*", "", caption).strip()


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(set(c.replace(":", "").replace("-", "")) == set() for c in cells):
            rows.append(cells)
        i += 1
    return rows, i


def table_spec(cols: int) -> str:
    if cols == 2:
        return r">{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.67\linewidth}"
    if cols == 3:
        return r">{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.29\linewidth}>{\raggedright\arraybackslash}p{0.34\linewidth}"
    if cols == 4:
        return r">{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}"
    if cols == 5:
        return r">{\raggedright\arraybackslash}p{0.20\linewidth}>{\centering\arraybackslash}p{0.15\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}>{\centering\arraybackslash}p{0.14\linewidth}"
    if cols == 6:
        return r">{\raggedright\arraybackslash}p{0.23\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.12\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.10\linewidth}"
    return "l" * cols


def make_table(rows: list[list[str]], caption: str, wide: bool = False) -> str:
    cols = len(rows[0])
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{inline_format(caption)}}}", r"\scriptsize"]
    lines.append(rf"\begin{{tabular}}{{{table_spec(cols)}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(r"\textbf{" + inline_format(cell) + "}" for cell in rows[0]) + r" \\")
    lines.append(r"\midrule")
    for row in rows[1:]:
        normalized = row + [""] * (cols - len(row))
        lines.append(" & ".join(inline_format(cell) for cell in normalized[:cols]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def table_caption(rows: list[list[str]], last_heading: str, index: int) -> str:
    header = [cell.lower() for cell in rows[0]]
    if header == ["setting", "purpose", "design"]:
        return "Benchmark difficulty regimes."
    if header[:4] == ["feature group", "easy", "overlap", "adaptive"]:
        return "Implementation-level difficulty controls."
    if header == ["model", "f1 mean +/- std", "recall mean +/- std"]:
        return "Multi-split stability results on the Overlap benchmark."
    if header == ["shifted component", "change"]:
        return "Generator-shift control changes."
    if header[:6] == ["model", "precision", "recall", "f1", "roc-auc", "pr-auc"] and last_heading == "Generator-Shift Robustness":
        return "Generator-shift robustness results."
    if header[:6] == ["cost scenario", "selected threshold", "precision", "recall", "false positives", "false negatives"]:
        return "Random Forest held-out test results using training-selected cost-sensitive thresholds."
    if header == ["case", "likely cause", "main feature pattern", "operational implication"]:
        return "Representative false-positive and false-negative patterns."
    if header[:6] == ["case", "true label", "predicted label", "likely simulated cause", "key feature pattern", "practical meaning"]:
        return "Representative error-case inspection results."
    names = {
        "Public Evidence and Benchmark Design Assumptions": "Public evidence and benchmark design assumptions.",
        "Difficulty Levels": "Benchmark difficulty regimes.",
        "OTPFloodGuard Benchmark Card": "OTPFloodGuard benchmark card.",
        "Assumption Replaceability": "Assumption replaceability matrix.",
        "Main Model Comparison": "Main Overlap simulated benchmark results.",
        "Difficulty Progression": "Difficulty progression results.",
        "Generator-Shift Robustness": "Generator-shift robustness results.",
        "Threshold Modes and Error Analysis": "Threshold modes and error analysis results.",
    }
    return names.get(last_heading, f"{last_heading} table {index}.")


def figure_latex(image_path: str, caption: str) -> str:
    cap = strip_fig_prefix(caption)
    filename = Path(image_path).name
    is_pipeline = filename == "benchmark_pipeline.png"
    env = "figure*" if is_pipeline else "figure"
    width = r"0.95\textwidth" if is_pipeline else r"\columnwidth"
    return "\n".join(
        [
            rf"\begin{{{env}}}[!t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{figures/{filename}}}",
            rf"\caption{{{inline_format(cap)}}}",
            rf"\end{{{env}}}",
        ]
    )


def reference_smart_quotes(text: str) -> str:
    """Use TeX opening/closing quotes for bibliography titles."""
    pieces: list[str] = []
    opening = True
    for char in text:
        if char == '"':
            pieces.append("``" if opening else "''")
            opening = not opening
        else:
            pieces.append(char)
    return "".join(pieces)


def references_to_bibitems(ref_lines: list[str]) -> str:
    items = [r"\begin{thebibliography}{00}"]
    for line in ref_lines:
        match = re.match(r"\[(\d+)\]\s*(.*)", line)
        if match:
            num, body = match.groups()
            body = body.rstrip().rstrip("\\")
            body = reference_smart_quotes(body)
            items.append(rf"\bibitem{{ref{num}}} {inline_format(body)}")
    items.append(r"\end{thebibliography}")
    return "\n".join(items)


def build_body(lines: list[str]) -> tuple[str, str, str]:
    abstract_parts: list[str] = []
    keywords = ""
    body: list[str] = []
    ref_lines: list[str] = []
    in_abstract = False
    in_refs = False
    before_abstract = True
    last_heading = ""
    table_index = 1
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if before_abstract and line != "## Abstract":
            i += 1
            continue
        if line == "## Abstract":
            before_abstract = False
            in_abstract = True
            i += 1
            continue
        if line.startswith("Keywords:"):
            keywords = line.replace("Keywords:", "").strip()
            in_abstract = False
            i += 1
            continue
        if line == "## References":
            in_refs = True
            i += 1
            continue
        if in_refs:
            if line:
                ref_lines.append(line)
            i += 1
            continue
        if in_abstract:
            if line:
                abstract_parts.append(line)
            i += 1
            continue
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            title = clean_heading(line[3:])
            last_heading = title
            body.append(rf"\section{{{inline_format(title)}}}")
            i += 1
            continue
        if line.startswith("### "):
            title = clean_heading(line[4:])
            last_heading = title
            body.append(rf"\subsection{{{inline_format(title)}}}")
            i += 1
            continue
        if line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                caption, image_path = m.groups()
                body.append(figure_latex(image_path, caption))
            i += 1
            continue
        if line.startswith("|"):
            rows, i = parse_table(lines, i)
            wide = len(rows[0]) >= 4 or last_heading in {
                "Public Evidence and Benchmark Design Assumptions",
                "Difficulty Levels",
                "OTPFloodGuard Benchmark Card",
                "Assumption Replaceability",
                "Main Model Comparison",
                "Generator-Shift Robustness",
            }
            body.append(make_table(rows, table_caption(rows, last_heading, table_index), wide=wide))
            table_index += 1
            continue
        if re.match(r"^\d+\.\s+", line):
            numbered: list[str] = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                numbered.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            body.append(r"\begin{enumerate}")
            for item in numbered:
                body.append(r"\item " + inline_format(item))
            body.append(r"\end{enumerate}")
            continue
        if line.startswith("- "):
            bullets: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                bullets.append(lines[i].strip()[2:])
                i += 1
            body.append(r"\begin{itemize}")
            for item in bullets:
                body.append(r"\item " + inline_format(item))
            body.append(r"\end{itemize}")
            continue
        if line.startswith("`") and line.endswith("`"):
            formula = line.strip("`").strip().strip("$")
            if "x'" in formula and "alpha" in formula:
                formula = r"x'_{i,f}=\left[n_{i,f}+\alpha\left(x_{i,f}-n_{i,f}\right)\right]\epsilon_{i,f},\quad \epsilon_{i,f}\sim\mathcal{N}(1,0.04^2)"
            else:
                formula = formula.replace("epsilon", r"\epsilon").replace("alpha", r"\alpha")
                formula = formula.replace("x'_f", r"x'_f").replace("x_f", r"x_f").replace("n_f", r"n_f")
            body.append("\n".join([r"\begin{equation}", formula, r"\end{equation}"]))
        else:
            body.append(inline_format(line))
        i += 1
    return "\n\n".join(abstract_parts), keywords, "\n\n".join(body + [references_to_bibitems(ref_lines)])


def main() -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError(
            "pdflatex was not found on PATH. Install a LaTeX distribution "
            "and ensure pdflatex is available from the command line."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    if TEMPLATE.exists():
        shutil.copy2(TEMPLATE, OUT / "IEEEtran.cls")
    fig_out = OUT / "figures"
    fig_out.mkdir(exist_ok=True)
    for fig in FIGURES.glob("*.png"):
        shutil.copy2(fig, fig_out / fig.name)

    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    abstract, keywords, body = build_body(lines)
    tex = rf"""\documentclass[conference]{{IEEEtran}}
\pdfobjcompresslevel=0
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{url}}
\usepackage{{microtype}}
\usepackage[hidelinks]{{hyperref}}
\hypersetup{{
  pdftitle={{OTPFloodGuard: A Public-Evidence-Constrained Benchmark for Lightweight OTP Flooding Detection}},
  pdfauthor={{Wenche An and Kamran Aziz}},
  pdfsubject={{A public-evidence-constrained simulated benchmark for lightweight OTP flooding detection}},
  pdfkeywords={{OTP flooding, SMS pumping, cybersecurity benchmark, simulated security data, machine learning}}
}}
\renewcommand{{\ttdefault}}{{cmtt}}
\setlength{{\textfloatsep}}{{6pt plus 1pt minus 1pt}}
\setlength{{\floatsep}}{{6pt plus 1pt minus 1pt}}
\setlength{{\intextsep}}{{6pt plus 1pt minus 1pt}}
\renewcommand{{\arraystretch}}{{1.08}}

\begin{{document}}

\title{{OTPFloodGuard: A Public-Evidence-Constrained Benchmark for Lightweight OTP Flooding Detection}}

\author{{\IEEEauthorblockN{{Wenche An$^{{1}}$, Kamran Aziz$^{{2}}$}}
\IEEEauthorblockA{{$^{{1}}$Computer Science\\
$^{{2}}$Digital Technologies\\
Hainan Bielefeld University of Applied Sciences, China\\
wenche.an.24@stu.hainan-biuh.edu.cn; kamran.aziz@hibiuh.edu.cn}}}}

\maketitle

\begin{{abstract}}
{inline_format(abstract)}
\end{{abstract}}

\begin{{IEEEkeywords}}
{inline_format(keywords)}
\end{{IEEEkeywords}}

{body}

\end{{document}}
"""
    tex_path = OUT / "OTPFloodGuard_ieee.tex"
    tex_path.write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=OUT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    generated_pdf = OUT / "OTPFloodGuard_ieee.pdf"
    target_pdf = ROOT / OUTPUT_PDF_NAME
    if generated_pdf.exists():
        shutil.copy2(generated_pdf, target_pdf)
    print(target_pdf)


if __name__ == "__main__":
    main()
