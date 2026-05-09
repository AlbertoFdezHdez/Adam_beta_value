from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "tables"


def read_tsv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fmt_num(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def latex_cell(value: str) -> str:
    return value.replace("%", r"\%")


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def selected_betas() -> str:
    rows = read_tsv("table_selected_betas.tsv")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Beta values selected by the refresh rule \(R_\beta=(1-\beta)T_{\mathrm{ES}}\approx1000\). The first block contains the development experiments used to calibrate \(R_0\), and the second block contains held-out validation experiments; gaps are validation-loss gaps relative to the oracle \(\beta^\star\).}",
        r"\label{tab:selected-betas}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cllcccc}",
        r"\toprule",
        r"Exp. & Model & Dataset & \(\beta^\star\) & \(T_{\mathrm{ES}}\) & selected \(\beta\) & gap (\%) \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i == 8:
            lines.append(r"\midrule")
        lines.append(
            f"{row['exp']} & {row['model']} & {row['dataset']} & "
            f"{fmt_num(row['beta_star'])} & {int(float(row['T_ES']))} & "
            f"{fmt_num(row['selected_beta'])} & {fmt_num(row['gap_pct'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines)


def metric_table(filename: str, caption: str, label: str, gap_col: str) -> str:
    rows = read_tsv(filename)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Development} ",
        r"& \multicolumn{3}{c}{Held-out} ",
        r"& \multicolumn{3}{c}{Global}",
        r"& \\",
        r"\cmidrule(lr){2-4}",
        r"\cmidrule(lr){5-7}",
        r"\cmidrule(lr){8-10}",
        r"Method ",
        r"& Mean & Max & CVaR ",
        r"& Mean & Max & CVaR ",
        r"& Mean & Max & CVaR ",
        r"& Gap \(<1\%\) \\",
        r"\midrule",
    ]
    for row in rows:
        method = row["method"]
        if method.startswith("sigma"):
            method = rf"\(\{method.replace('=', '=')}\)"
        elif method == "Refresh R0=1000":
            method = r"Refresh \(R_0=1000\)"
        elif method == "Fixed beta=0.944":
            method = r"Fixed \(\beta=0.944\)"
        cells = {
            "dev_mean": fmt_num(row["dev_mean"]),
            "dev_max": fmt_num(row["dev_max"]),
            "dev_cvar": fmt_num(row["dev_cvar"]),
            "held_mean": fmt_num(row["held_mean"]),
            "held_max": fmt_num(row["held_max"]),
            "held_cvar": fmt_num(row["held_cvar"]),
            "global_mean": fmt_num(row["global_mean"]),
            "global_max": fmt_num(row["global_max"]),
            "global_cvar": fmt_num(row["global_cvar"]),
            "gap": latex_cell(row[gap_col]),
        }
        if filename == "table_main_results.tsv":
            if row["method"] == "Fixed beta=0.944":
                cells["dev_mean"] = rf"\textbf{{{cells['dev_mean']}}}"
                cells["global_mean"] = rf"\textbf{{{cells['global_mean']}}}"
            elif row["method"] == "Refresh R0=1000":
                for key in ("dev_max", "dev_cvar", "held_mean", "held_max", "held_cvar", "global_max", "global_cvar", "gap"):
                    cells[key] = rf"\textbf{{{cells[key]}}}"
        lines += [
            method,
            f"& {cells['dev_mean']} & {cells['dev_max']} & {cells['dev_cvar']}",
            f"& {cells['held_mean']} & {cells['held_max']} & {cells['held_cvar']}",
            f"& {cells['global_mean']} & {cells['global_max']} & {cells['global_cvar']}",
            f"& {cells['gap']} \\\\",
            "",
        ]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines)


def stability_intervals() -> str:
    rows = read_tsv("table_beta_stability_intervals.tsv")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Stability intervals of the beta selected by the refresh rule \(R_\beta=(1-\beta)T_{\mathrm{ES}}\approx1000\). For each experiment, \(T_{\mathrm{ES}}\) denotes the rounded horizon used by the rule, and the last two columns show the range of \(T_{\mathrm{ES}}\) values for which the selected grid beta remains unchanged.}",
        r"\label{tab:beta-stability-intervals}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cllcccc}",
        r"\toprule",
        r"Exp. & Model & Dataset & selected \(\beta\) & \(T_{\mathrm{ES}}\) & lower \(T_{\mathrm{ES}}\) & upper \(T_{\mathrm{ES}}\) \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i == 8:
            lines.append(r"\midrule")
        lines.append(
            f"{row['exp']} & {row['model']} & {row['dataset']} & "
            f"{fmt_num(row['selected_beta'])} & {int(float(row['T_ES']))} & "
            f"{int(float(row['lower_T_ES']))} & {int(float(row['upper_T_ES']))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write(OUT_DIR / "table_selected_betas.tex", selected_betas())
    write(
        OUT_DIR / "table_main_results.tex",
        metric_table(
            "table_main_results.tsv",
            r"Relative gaps to the per-experiment oracle beta. The refresh rule \(R_0=1000\) selects one beta per experiment using \(T_{\mathrm{ES}}\). The development block contains the 8 experiments used to calibrate \(R_0\), while the held-out block contains the 3 remaining experiments.",
            "tab:main-results",
            "gap_lt_1",
        ),
    )
    write(OUT_DIR / "table_beta_stability_intervals.tex", stability_intervals())
    write(
        OUT_DIR / "table_refresh_noise_robustness.tex",
        metric_table(
            "table_refresh_noise_robustness.tsv",
            r"Robustness of the refresh rule under multiplicative noise in \(T_{\mathrm{ES}}\). For each experiment, we perturb the estimated horizon as \(T'=T_{\mathrm{ES}}(1+\epsilon)\), with \(\epsilon\sim\mathcal{N}(0,\sigma)\), and select the closest grid beta induced by the refresh rule. For each \(\sigma>0\), results are averaged over 20 perturbations per experiment.",
            "tab:refresh-noise-robustness",
            "gap_lt_1_pct",
        ),
    )
    for path in sorted(OUT_DIR.glob("*.tex")):
        print(path)


if __name__ == "__main__":
    main()
