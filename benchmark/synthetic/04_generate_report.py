import csv
import json
import os
from html import escape
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")
DETAIL_CSV = os.path.join(RESULTS_DIR, "detail.csv")
REPORT_DATA = os.path.join(RESULTS_DIR, "report_data.json")
OUTPUT_HTML = os.path.join(RESULTS_DIR, "report.html")


def load_detail():
    rows = []
    with open(DETAIL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_report_data():
    with open(REPORT_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def get_readers(detail_rows):

    sample = detail_rows[0]
    readers = []
    for key in sample:
        if key.endswith("_correct"):
            readers.append(key.replace("_correct", ""))
    return readers

SCENARIO_META = {
    "A01_doc_center": ("A. Placement and scale", "Document center"),
    "A02_doc_corner_small": ("A. Placement and scale", "Small code at bottom right"),
    "A03_doc_top_right": ("A. Placement and scale", "Document top right"),
    "A04_doc_with_table": ("A. Placement and scale", "Document with a table"),
    "A05_on_desk": ("A. Placement and scale", "Document on a desk"),
    "A06_tiny_qr": ("A. Placement and scale", "2 cm code in an A4 corner"),
    "B01_shadow_left": ("B. Lighting and degradation", "Left shadow"),
    "B02_shadow_right": ("B. Lighting and degradation", "Right shadow"),
    "B03_shadow_diagonal": ("B. Lighting and degradation", "Diagonal shadow"),
    "B04_shadow_top": ("B. Lighting and degradation", "Top shadow"),
    "B05_spotlight": ("B. Lighting and degradation", "Spotlight"),
    "B06_spotlight_offset": ("B. Lighting and degradation", "Off-center spotlight"),
    "B07_paper_texture": ("B. Lighting and degradation", "Paper texture and wrinkles"),
    "B08_stained": ("B. Lighting and degradation", "Stains"),
    "B09_low_contrast": ("B. Lighting and degradation", "Low-contrast print"),
    "B10_very_low_contrast": ("B. Lighting and degradation", "Very low-contrast print"),
    "B11_inverted": ("B. Lighting and degradation", "Inverted polarity"),
    "B12_colored_paper": ("B. Lighting and degradation", "Colored paper"),
    "B13_heavy_noise": ("B. Lighting and degradation", "Heavy noise"),
    "C01_persp_mild": ("C. Distortion and rotation", "Mild perspective distortion"),
    "C02_persp_moderate": ("C. Distortion and rotation", "Moderate perspective distortion"),
    "C03_persp_strong": ("C. Distortion and rotation", "Strong perspective distortion"),
    "C04_persp_extreme": ("C. Distortion and rotation", "Extreme perspective distortion"),
    "C05_topdown_20deg": ("C. Distortion and rotation", "Top view at 20°"),
    "C06_topdown_40deg": ("C. Distortion and rotation", "Top view at 40°"),
    "C07_topdown_55deg": ("C. Distortion and rotation", "Top view at 55°"),
    "C08_side_20deg": ("C. Distortion and rotation", "Side view at 20°"),
    "C09_side_40deg": ("C. Distortion and rotation", "Side view at 40°"),
    "C10_barrel": ("C. Distortion and rotation", "Barrel distortion"),
    "C11_barrel_strong": ("C. Distortion and rotation", "Strong barrel distortion"),
    "C12_pincushion": ("C. Distortion and rotation", "Pincushion distortion"),
    "C13_wave": ("C. Distortion and rotation", "Wave distortion"),
    "C14_wave_strong": ("C. Distortion and rotation", "Strong wave distortion"),
    "C15_cylindrical": ("C. Distortion and rotation", "Cylindrical distortion"),
    "C16_cylindrical_tight": ("C. Distortion and rotation", "Tight cylindrical distortion"),
    "C17_rotate_5": ("C. Distortion and rotation", "Rotation by 5°"),
    "C18_rotate_15": ("C. Distortion and rotation", "Rotation by 15°"),
    "C19_rotate_45": ("C. Distortion and rotation", "Rotation by 45°"),
    "C20_rotate_90": ("C. Distortion and rotation", "Rotation by 90°"),
    "C21_rotate_180": ("C. Distortion and rotation", "Rotation by 180°"),
    "D01_phone_good": ("D. Capture conditions", "Good phone capture"),
    "D02_phone_medium": ("D. Capture conditions", "Moderate phone capture"),
    "D03_phone_bad": ("D. Capture conditions", "Poor phone capture"),
    "D04_motion_blur_light": ("D. Capture conditions", "Light motion blur"),
    "D05_motion_blur_heavy": ("D. Capture conditions", "Heavy motion blur"),
    "D06_jpeg_q15": ("D. Capture conditions", "JPEG Q15"),
    "D07_jpeg_q5": ("D. Capture conditions", "JPEG Q5"),
    "D08_blur_gaussian": ("D. Capture conditions", "Gaussian blur"),
    "D09_downscale_30pct": ("D. Capture conditions", "Downscaled to 30%"),
    "D10_downscale_15pct": ("D. Capture conditions", "Downscaled to 15%"),
    "E01_desk_persp": ("E. Combined conditions", "Desk and perspective"),
    "E02_desk_topdown": ("E. Combined conditions", "Desk and top view"),
    "E03_shadow_persp": ("E. Combined conditions", "Shadow and perspective"),
    "E04_texture_barrel": ("E. Combined conditions", "Paper texture and barrel distortion"),
    "E05_noise_persp": ("E. Combined conditions", "Noise and perspective"),
    "E06_phone_persp": ("E. Combined conditions", "Phone capture and perspective"),
    "E07_shadow_wave": ("E. Combined conditions", "Shadow and wave distortion"),
    "E08_combined_hard": ("E. Combined conditions", "Hard combined conditions"),
    "E09_combined_extreme": ("E. Combined conditions", "Extreme combined conditions"),
    "E10_cylinder_shadow": ("E. Combined conditions", "Cylinder, shadow, and desk"),
    "F01_multi_two": ("F. Multiple QR codes", "Two QR codes"),
    "F02_multi_three": ("F. Multiple QR codes", "Three QR codes"),
    "F03_multi_sizes": ("F. Multiple QR codes", "Mixed code sizes"),
    "F04_multi_desk_persp": ("F. Multiple QR codes", "Multiple codes, desk, and perspective"),
    "F05_multi_persp_strong": ("F. Multiple QR codes", "Multiple codes and strong perspective"),
}

QR_DATA_INFO = {
    "short": ("OK", 2, "V1"),
    "url": ("URL", 33, "V3"),
    "wifi": ("WiFi", 38, "V3"),
    "uuid": ("UUID", 36, "V3"),
    "numeric": ("40 digits", 40, "V2"),
    "vcard": ("vCard", 66, "V5"),
    "long_url": ("Long URL", 92, "V6"),
    "japanese": ("Japanese text", 26, "V5"),
    "dense_300": ("JSON (265 chars)", 265, "V12"),
    "dense_jp_500": ("Japanese text (261 chars)", 261, "V22"),
    "dense_600": ("English text (572 chars)", 572, "V19"),
    "dense_1200": ("Event data (1113 chars)", 1113, "V27"),
}


def cell_class(correct_count, total):
    if correct_count == total:
        return "p"
    if correct_count == 0:
        return "z"
    if correct_count >= total * 0.7:
        return "g"
    return "w"


def build_scenario_table_rows(detail_rows, readers, scenario_names):

    scenario_stats = {}
    for sname in scenario_names:
        relevant = [r for r in detail_rows if r["scenario"] == sname]
        total = len(relevant)
        stats = {}
        for rd in readers:
            correct = sum(1 for r in relevant if r[f"{rd}_correct"] == "1")
            stats[rd] = (correct, total)
        scenario_stats[sname] = stats
    return scenario_stats


def build_ng_gallery(detail_rows, readers):

    # ng_data[scenario][reader] = [(filename, data_name, expected_short, decoded_short), ...]
    ng_data = defaultdict(lambda: defaultdict(list))
    for row in detail_rows:
        scenario = row["scenario"]
        filename = row["filename"]
        data_name = row["data_name"]
        for rd in readers:
            if row[f"{rd}_correct"] == "0":
                decoded = row.get(f"{rd}_decoded", "")
                if len(decoded) > 60:
                    decoded = decoded[:57] + "..."
                expected_short = row["expected"]
                if len(expected_short) > 60:
                    expected_short = expected_short[:57] + "..."
                ng_data[scenario][rd].append((filename, data_name, expected_short, decoded))
    return ng_data


def generate_html(detail_rows, report_data):
    readers = get_readers(detail_rows)
    total_images = report_data["total_images"]
    summary = report_data["summary"]
    scenario_names = sorted(set(r["scenario"] for r in detail_rows))
    scenario_stats = build_scenario_table_rows(detail_rows, readers, scenario_names)
    ng_data = build_ng_gallery(detail_rows, readers)

    # Group results by payload type
    data_names = sorted(set(r["data_name"] for r in detail_rows))
    data_stats = {}
    for dn in data_names:
        relevant = [r for r in detail_rows if r["data_name"] == dn]
        total = len(relevant)
        stats = {}
        for rd in readers:
            correct = sum(1 for r in relevant if r[f"{rd}_correct"] == "1")
            stats[rd] = (correct, total)
        data_stats[dn] = stats

    html = []
    html.append(HTML_HEAD)

    # --- Recognition pass rate ---
    html.append('<h2>Recognition pass rate</h2>')
    html.append('<p class="hint">qore timings cover detection and decoding on in-memory images. Third-party synthetic baseline timings include image loading and decompression.</p>')
    html.append('<div class="cards">')
    rank_colors = ["var(--green)", "var(--accent)", "var(--orange)", "var(--red)"]
    rank_classes = ["r1", "r2", "r3", "r4"]
    for i, s in enumerate(summary):
        pct = s["Pass rate (%)"]
        html.append(f'''<div class="card {rank_classes[i % len(rank_classes)]}">
  <div class="badge">#{i+1}</div>
  <div class="nm">{s["Engine"]}</div>
  <div class="acc" style="color:{rank_colors[i % len(rank_colors)]}">{pct}%</div>
  <div class="acc-s">{s["Passed"]} / {s["Images"]} passed</div>
  <div class="bar-bg"><div class="bar" style="width:{pct}%;background:{rank_colors[i % len(rank_colors)]}"></div></div>
  <div class="st"><span>Mean {s["Mean time (ms)"]} ms/image</span><span>Total {float(s["Total time (ms)"]) / 1000:.2f} s</span></div>
</div>''')
    html.append('</div>')

    html.append('<h2>Pass counts and failures by scenario</h2>')
    html.append('<p class="hint">Select a scenario row to view failed samples.</p>')

    current_category = None
    for sname in scenario_names:
        meta = SCENARIO_META.get(sname, ("", sname))
        category, desc = meta

        if category != current_category:
            if current_category is not None:
                html.append('</tbody></table>')
            current_category = category
            html.append(f'<h3>{category}</h3>')
            html.append('<table class="scenario-tbl"><thead><tr><th>Scenario</th><th>Description</th>')
            for rd in readers:
                html.append(f'<th>{rd}</th>')
            html.append('</tr></thead><tbody>')

        stats = scenario_stats[sname]

        has_ng = any(stats[rd][0] < stats[rd][1] for rd in readers)

        html.append(f'<tr class="scenario-row{"" if not has_ng else " has-ng"}" data-scenario="{sname}">')
        html.append(f'<td>{sname.split("_", 1)[0]}</td><td>{desc}</td>')
        for rd in readers:
            c, t = stats[rd]
            cls = cell_class(c, t)
            html.append(f'<td class="{cls}">{c}/{t}</td>')
        html.append('</tr>')

        if has_ng:
            html.append(f'<tr class="ng-row" id="ng-{sname}"><td colspan="{2 + len(readers)}">')
            html.append(f'<div class="ng-container">')

            for rd in readers:
                c, t = stats[rd]
                if c == t:
                    continue
                ng_items = ng_data.get(sname, {}).get(rd, [])
                if not ng_items:
                    continue

                html.append(f'<div class="ng-reader-section">')
                html.append(f'<div class="ng-reader-label"><span class="ng-badge">{rd}</span> {t - c} failed</div>')
                html.append('<div class="ng-grid">')
                for filename, data_name, expected, decoded in ng_items:
                    img_path = f"../test_images/{filename}"
                    di = QR_DATA_INFO.get(data_name, (data_name, "", ""))
                    html.append(f'''<div class="ng-card">
  <div class="ng-img-wrap"><img loading="lazy" src="{img_path}" alt="{filename}"></div>
  <div class="ng-info">
    <div class="ng-filename">{di[0]} ({di[2]})</div>
    <div class="ng-detail">Decoded: {escape(decoded.replace(chr(0), "�")) if decoded else "<em>No decoded payload</em>"}</div>
  </div>
</div>''')
                html.append('</div></div>')

            html.append('</div></td></tr>')

    html.append('</tbody></table>')

    html.append('<h2>Pass counts by payload type</h2>')
    html.append('<div class="two-col"><div><h3>Short payloads</h3><table><thead><tr><th>Payload</th><th>Characters</th><th>Ver.</th>')
    for rd in readers:
        html.append(f'<th>{rd}</th>')
    html.append('</tr></thead><tbody>')

    light = [dn for dn in data_names if not dn.startswith("dense")]
    dense = [dn for dn in data_names if dn.startswith("dense")]

    for dn in light:
        di = QR_DATA_INFO.get(dn, (dn, "?", "?"))
        stats = data_stats[dn]
        html.append(f'<tr><td>{di[0]}</td><td>{di[1]}</td><td>{di[2]}</td>')
        for rd in readers:
            c, t = stats[rd]
            cls = cell_class(c, t)
            html.append(f'<td class="{cls}">{c}/{t}</td>')
        html.append('</tr>')

    html.append('</tbody></table></div><div><h3>Long payloads</h3><table><thead><tr><th>Payload</th><th>Characters</th><th>Ver.</th>')
    for rd in readers:
        html.append(f'<th>{rd}</th>')
    html.append('</tr></thead><tbody>')

    for dn in dense:
        di = QR_DATA_INFO.get(dn, (dn, "?", "?"))
        stats = data_stats[dn]
        html.append(f'<tr><td>{di[0]}</td><td>{di[1]}</td><td>{di[2]}</td>')
        for rd in readers:
            c, t = stats[rd]
            cls = cell_class(c, t)
            html.append(f'<td class="{cls}">{c}/{t}</td>')
        html.append('</tr>')

    html.append('</tbody></table></div></div>')

    html.append(HTML_FOOTER)

    return "\n".join(html)


# ============================================================

# ============================================================

HTML_HEAD = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QR Code Recognition Benchmark</title>
<style>
  :root{--bg:#f8f9fa;--card:#fff;--border:#dee2e6;--text:#212529;--muted:#6c757d;--accent:#0d6efd;--green:#198754;--yellow:#ffc107;--orange:#fd7e14;--red:#dc3545}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:"Segoe UI",sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:24px;max-width:1300px;margin:0 auto}
  h1{font-size:1.55rem;margin-bottom:2px}
  .sub{color:var(--muted);font-size:.88rem;margin-bottom:20px}
  h2{font-size:1.15rem;margin:30px 0 10px;padding-bottom:5px;border-bottom:2px solid var(--accent);display:inline-block}
  h3{font-size:.97rem;margin:20px 0 6px;color:var(--muted)}
  .hint{font-size:.8rem;color:var(--muted);margin-bottom:8px}

  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:6px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;position:relative}
  .card.r1{border-left:5px solid var(--green)}.card.r2{border-left:5px solid var(--accent)}
  .card.r3{border-left:5px solid var(--orange)}.card.r4{border-left:5px solid var(--muted)}
  .card .badge{position:absolute;top:10px;right:12px;font-size:1.7rem;font-weight:800;opacity:.12}
  .card .nm{font-size:1.05rem;font-weight:700;margin-bottom:6px}
  .card .acc{font-size:2.1rem;font-weight:800;line-height:1}
  .card .acc-s{font-size:.82rem;color:var(--muted);margin-top:2px}
  .card .bar-bg{margin-top:8px;height:7px;background:#e9ecef;border-radius:4px;overflow:hidden}
  .card .bar{height:100%;border-radius:4px}
  .card .st{margin-top:10px;font-size:.82rem;color:var(--muted)}.card .st span{margin-right:12px}

  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;font-size:.84rem;margin-bottom:6px}
  th,td{padding:6px 10px;text-align:center;border-bottom:1px solid #f0f0f0}
  th{background:#f1f3f5;font-weight:600;font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.02em}
  td:first-child,th:first-child{text-align:left;font-weight:500}
  tr:last-child td{border-bottom:none}

  .p{background:#d1e7dd;color:#0f5132;font-weight:600}
  .g{background:#fff3cd;color:#664d03}
  .w{background:#f8d7da;color:#842029}
  .z{background:#e2e3e5;color:#6c757d;font-weight:600}

  /* scenario row click */
  .scenario-row{cursor:default;transition:background .15s}
  .scenario-row.has-ng{cursor:pointer}
  .scenario-row.has-ng:hover{background:#e9ecef}
  .scenario-row.has-ng td:first-child::before{content:"▶ ";font-size:.7rem;color:var(--muted);transition:transform .2s}
  .scenario-row.has-ng.open td:first-child::before{content:"▼ "}

  /* NG row */
  .ng-row{display:none}
  .ng-row.open{display:table-row}
  .ng-row>td{padding:0;background:#fafbfc;border-top:1px dashed var(--border)}
  .ng-container{padding:12px 16px;max-height:600px;overflow-y:auto}

  .ng-reader-section{margin-bottom:14px}
  .ng-reader-section:last-child{margin-bottom:0}
  .ng-reader-label{font-size:.82rem;font-weight:600;margin-bottom:6px;display:flex;align-items:center;gap:6px}
  .ng-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.72rem;font-weight:700;background:#f8d7da;color:#842029}

  .ng-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
  .ng-card{background:var(--card);border:1px solid var(--border);border-radius:6px;overflow:hidden;transition:box-shadow .15s}
  .ng-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.1)}
  .ng-img-wrap{width:100%;aspect-ratio:3/4;overflow:hidden;background:#f0f0f0;display:flex;align-items:center;justify-content:center}
  .ng-img-wrap img{width:100%;height:100%;object-fit:contain;cursor:zoom-in}
  .ng-info{padding:6px 8px}
  .ng-filename{font-size:.78rem;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ng-detail{font-size:.7rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}

  /* lightbox */
  .lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;align-items:center;justify-content:center;cursor:zoom-out}
  .lightbox.show{display:flex}
  .lightbox img{max-width:92vw;max-height:92vh;object-fit:contain;border-radius:6px;box-shadow:0 4px 24px rgba(0,0,0,.5)}
  .lightbox .lb-caption{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);color:#fff;font-size:.85rem;background:rgba(0,0,0,.6);padding:4px 14px;border-radius:6px}

  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:800px){.two-col{grid-template-columns:1fr}}

  .analysis{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:22px;margin-top:14px;line-height:1.85}
  .analysis h3{margin-top:16px;margin-bottom:3px;color:var(--text);font-size:.97rem}
  .analysis h3:first-child{margin-top:0}
  .analysis ul{margin:3px 0 8px 18px}.analysis li{margin-bottom:3px}
  .verdict{margin-top:16px;padding:16px;background:linear-gradient(135deg,#e7f1ff,#f0f7ff);border-radius:8px;border-left:4px solid var(--accent)}
  .verdict strong{color:var(--accent)}
  .note{margin-top:20px;padding:10px 14px;background:#fff3cd;border-radius:8px;font-size:.83rem;color:#664d03}
  footer{margin-top:28px;padding-top:10px;border-top:1px solid var(--border);font-size:.78rem;color:var(--muted);text-align:center}
</style>
</head>
<body>

<h1>QR Code Recognition Benchmark</h1>
<p class="sub"><strong>780 synthetic images</strong> · 65 scenarios · 12 payload types</p>

<!-- lightbox -->
<div class="lightbox" id="lightbox" onclick="this.classList.remove('show')">
  <img id="lb-img" src="" alt="">
  <div class="lb-caption" id="lb-cap"></div>
</div>
'''

HTML_FOOTER = r'''
<footer>Generated by qrcode-reader-benchmark &mdash; 2026-03-09</footer>

<script>

document.querySelectorAll('.scenario-row.has-ng').forEach(row => {
  row.addEventListener('click', () => {
    const sname = row.dataset.scenario;
    const ngRow = document.getElementById('ng-' + sname);
    if (ngRow) {
      row.classList.toggle('open');
      ngRow.classList.toggle('open');
    }
  });
});

document.addEventListener('click', e => {
  if (e.target.matches('.ng-img-wrap img')) {
    e.stopPropagation();
    const lb = document.getElementById('lightbox');
    document.getElementById('lb-img').src = e.target.src;
    document.getElementById('lb-cap').textContent = e.target.alt;
    lb.classList.add('show');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('show');
});
</script>
</body>
</html>
'''


def add_qore_results(detail_rows, report_data):
    results_dir = os.path.join(os.path.dirname(RESULTS_DIR), "..", "results")
    with open(os.path.join(results_dir, "comparison_data.json"), encoding="utf-8") as f:
        qore = json.load(f)["synthetic"]["engines"]["qore"]
    with open(os.path.join(results_dir, "qore_synthetic_failures.csv"), newline="", encoding="utf-8") as f:
        failures = {row["filename"]: row for row in csv.DictReader(f)}

    filenames = {row["filename"] for row in detail_rows}
    correct, total = qore["correct"], qore["total"]
    if not failures.keys() <= filenames or total != len(filenames) or total - correct != len(failures):
        raise ValueError("qore failure details do not match the summary")
    for row in detail_rows:
        failure = failures.get(row["filename"])
        row["qore_correct"] = "0" if failure else "1"
        row["qore_decoded"] = failure["decoded"] if failure else ""
    report_data["summary"].insert(0, {
        "Engine": "qore",
        "Passed": correct,
        "Images": total,
        "Pass rate (%)": f"{correct / total:.2%}".rstrip("%"),
        "Total time (ms)": qore["total_ms"],
        "Mean time (ms)": f"{qore['avg_ms']:.2f}",
    })


def main():
    detail_rows = load_detail()
    report_data = load_report_data()
    add_qore_results(detail_rows, report_data)
    html = generate_html(detail_rows, report_data)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report: {OUTPUT_HTML}")
    print(f"  {len(detail_rows)} test results, {len(get_readers(detail_rows))} readers")


if __name__ == "__main__":
    main()
