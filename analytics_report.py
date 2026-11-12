"""Build a self-contained HTML dashboard from the SQLite analytics database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


def _json_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def load_report_data(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        runs: list[dict[str, Any]] = []
        for run in db.execute("SELECT * FROM runs ORDER BY run_id").fetchall():
            run_id = int(run["run_id"])
            label = str(run["architecture"] or "unknown")
            architecture, separator, scenario = label.partition(":")
            scenario = scenario if separator else "live"
            ticks = [
                dict(row)
                for row in db.execute(
                    """
                    SELECT history_index,tick,active_count,completed_count,rejected_count,
                           total_generated,next_input_idx,is_current
                    FROM ticks WHERE run_id=? ORDER BY history_index
                    """,
                    (run_id,),
                ).fetchall()
            ]
            events = [
                {"event_type": row["event_type"], "count": int(row["count"])}
                for row in db.execute(
                    "SELECT event_type,COUNT(*) AS count FROM events WHERE run_id=? GROUP BY event_type ORDER BY count DESC,event_type",
                    (run_id,),
                ).fetchall()
            ]
            metric_row = db.execute("SELECT * FROM run_metrics WHERE run_id=?", (run_id,)).fetchone()
            current_index = next((int(row["history_index"]) for row in ticks if row["is_current"]), None)
            if current_index is None and ticks:
                current_index = int(ticks[-1]["history_index"])
            outcomes = []
            neurons = []
            if current_index is not None:
                outcomes = [
                    {"status": row["status_code"] or "UNKNOWN", "count": int(row["count"])}
                    for row in db.execute(
                        "SELECT status_code,COUNT(*) AS count FROM requests WHERE run_id=? AND history_index=? GROUP BY status_code ORDER BY count DESC",
                        (run_id, current_index),
                    ).fetchall()
                ]
                neurons = [
                    dict(row)
                    for row in db.execute(
                        """
                        SELECT row,col,state_code,is_operational,pending_fault,
                               processed_requests_count,unprocessed_requests_count,request_id
                        FROM neuron_states WHERE run_id=? AND history_index=? ORDER BY row,col
                        """,
                        (run_id, current_index),
                    ).fetchall()
                ]

            current = next((row for row in ticks if row["is_current"]), ticks[-1] if ticks else {})
            generated = int((metric_row["total_generated"] if metric_row else None) or current.get("total_generated") or 0)
            completed = int((metric_row["completed"] if metric_row else None) or current.get("completed_count") or 0)
            rejected = int((metric_row["rejected"] if metric_row else None) or current.get("rejected_count") or 0)
            faulty = sum(1 for neuron in neurons if neuron.get("state_code") == "FAULTY")
            pending = sum(1 for neuron in neurons if neuron.get("pending_fault"))
            runs.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "architecture": architecture,
                    "scenario": scenario,
                    "topology": json.loads(run["topology_json"]),
                    "created_at": run["created_at"],
                    "finished_at": run["finished_at"],
                    "failure_mode": run["failure_mode"] or "none",
                    "failure_target_count": run["failure_target_count"] or 0,
                    "failure_probability": run["failure_probability"] if run["failure_probability"] is not None else 1.0,
                    "failure_seed": run["failure_seed"],
                    "metrics": dict(metric_row) if metric_row else {},
                    "ticks": ticks,
                    "events": events,
                    "outcomes": outcomes,
                    "neurons": neurons,
                    "current": {
                        "tick": int(current.get("tick") or 0),
                        "generated": generated,
                        "completed": completed,
                        "rejected": rejected,
                        "active": int(current.get("active_count") or 0),
                        "faulty": faulty,
                        "pending": pending,
                        "success_rate": float(metric_row["success_rate"]) if metric_row else (round(completed * 100 / generated, 1) if generated else 0.0),
                        "mean_duration": metric_row["mean_duration_ticks"] if metric_row else None,
                        "p95_duration": metric_row["p95_duration_ticks"] if metric_row else None,
                        "completed_with_faults": metric_row["completed_with_faults"] if metric_row else 0,
                    },
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(path),
        "runs": runs,
    }


def render_report(data: dict[str, Any]) -> str:
    payload = _json_script(data)
    return f'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>НС АЦП · аналитика</title>
<style>
:root {{ color-scheme: dark; --bg:#1a1a1a; --panel:#21252b; --panel2:#2b2b2b; --line:#3c3c3c; --text:#dcdcdc; --muted:#9e9e9e; --cyan:#4fc3f7; --green:#66bb6a; --red:#ef5350; --yellow:#ffca28; --purple:#ab47bc; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 "Segoe UI",system-ui,sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:28px 32px 48px; }} header {{ display:flex; justify-content:space-between; gap:20px; align-items:end; margin-bottom:24px; }}
h1 {{ margin:0; font-size:34px; letter-spacing:-.03em; }} h2 {{ margin:0 0 14px; font-size:17px; }} p {{ color:var(--muted); margin:6px 0 0; }}
.toolbar,.cards,.grid {{ display:grid; gap:12px; }} .toolbar {{ grid-template-columns:repeat(3,minmax(180px,1fr)) auto; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; margin-bottom:14px; align-items:end; }}
label {{ display:grid; gap:5px; color:var(--muted); font-size:12px; }} select,button {{ border:1px solid var(--line); border-radius:8px; background:var(--panel2); color:var(--text); padding:9px 10px; }} button {{ cursor:pointer; color:var(--cyan); }} button:hover {{ border-color:var(--cyan); }}
.cards {{ grid-template-columns:repeat(6,minmax(110px,1fr)); margin-bottom:14px; }} .card {{ background:linear-gradient(145deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:14px; padding:14px; }} .card b {{ display:block; font-size:24px; margin-bottom:3px; }} .card span {{ color:var(--muted); font-size:12px; }}
.grid {{ grid-template-columns:1.2fr .8fr; margin-bottom:14px; }} section.panel {{ background:rgba(18,25,35,.94); border:1px solid var(--line); border-radius:14px; padding:16px; min-width:0; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }} th,td {{ text-align:left; padding:9px 7px; border-bottom:1px solid #323842; white-space:nowrap; }} th {{ color:var(--muted); font-weight:500; }} tr:hover td {{ background:#2c313a; }} .num {{ text-align:right; }}
.barrow {{ display:grid; grid-template-columns:75px 1fr 42px; gap:8px; align-items:center; margin:9px 0; }} .barlabel {{ color:var(--muted); }} .bar {{ height:10px; background:#323842; border-radius:99px; overflow:hidden; }} .bar i {{ display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--cyan),var(--purple)); }}
svg {{ width:100%; height:280px; display:block; background:#1a1a1a; border-radius:10px; border:1px solid #323842; }} .empty {{ color:var(--muted); padding:40px 8px; text-align:center; }}
.heat {{ display:grid; gap:4px; align-items:stretch; }} .cell {{ min-height:42px; border:1px solid #455464; border-radius:6px; display:grid; place-items:center; font-size:11px; color:#f4f7fa; position:relative; }} .cell small {{ color:#b5c2ce; font-size:9px; }} .fault {{ background:#8e3542; }} .busy {{ background:#006d77; }} .done {{ background:#46515b; }} .free {{ background:#26313a; }} .pending {{ outline:2px solid var(--red); outline-offset:-2px; }}
footer {{ color:var(--muted); font-size:12px; padding-top:6px; }} .pill {{ display:inline-block; padding:3px 7px; border-radius:99px; background:#243448; color:#b9dfff; }}
@media(max-width:1000px) {{ main {{ padding:20px 14px; }} header {{ display:block; }} .toolbar,.cards,.grid {{ grid-template-columns:1fr 1fr; }} .cards {{ grid-template-columns:repeat(3,1fr); }} }}
@media(max-width:620px) {{ .toolbar,.grid,.cards {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:26px; }} section.panel {{ overflow:auto; }} }}
</style>
</head>
<body>
<main>
<header><div><h1>НС АЦП · аналитика экспериментов</h1><p>Сравнение архитектур, отказов, веток истории и потактовой динамики.</p></div><span class="pill" id="dbName"></span></header>
<div class="toolbar">
<label>Архитектура<select id="architecture"></select></label>
<label>Сценарий<select id="scenario"></select></label>
<label>Запуск для графиков<select id="run"></select></label>
<button id="reset">Сбросить фильтры</button>
</div>
<div class="cards" id="cards"></div>
<div class="grid"><section class="panel"><h2>Сравнение запусков</h2><div id="comparison"></div></section><section class="panel"><h2>События выбранного запуска</h2><div id="events"></div></section></div>
<div class="grid"><section class="panel"><h2>Динамика выбранного запуска</h2><svg id="chart" viewBox="0 0 800 280" preserveAspectRatio="none"></svg></section><section class="panel"><h2>Состояние нейронов выбранного запуска</h2><div id="heatmap"></div></section></div>
<footer>Сформировано <span id="generated"></span>. Для обновления запусти <code>python analytics_report.py</code>.</footer>
</main>
<script>
const DATA = {payload};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const uniq = xs => [...new Set(xs)].sort();
let selectedRun = null;
function options(select, values, current, labeler=x=>x) {{ select.innerHTML = values.map(x => `<option value="${{esc(x)}}">${{esc(labeler(x))}}</option>`).join(''); if (current != null) select.value=String(current); }}
function filteredRuns() {{ const a=$('architecture').value, s=$('scenario').value; return DATA.runs.filter(r => (!a || r.architecture===a) && (!s || r.scenario===s)); }}
function renderFilters() {{
  const architectures=uniq(DATA.runs.map(r=>r.architecture)); const scenarios=uniq(DATA.runs.map(r=>r.scenario));
  options($('architecture'),['',...architectures],$('architecture').value,x=>x||'Все архитектуры'); options($('scenario'),['',...scenarios],$('scenario').value,x=>x||'Все сценарии');
  const runs=filteredRuns(); if(!runs.some(r=>String(r.run_id)===String(selectedRun))) selectedRun=runs[0]?.run_id ?? null;
  options($('run'),runs.map(r=>r.run_id),selectedRun,id=>{{const r=DATA.runs.find(x=>x.run_id===Number(id)); return r?`#${{r.run_id}} · ${{r.label}}`:id;}});
}}
function selected() {{ return DATA.runs.find(r=>r.run_id===Number($('run').value)) || filteredRuns()[0]; }}
function renderCards(runs) {{
  const sum=(key)=>runs.reduce((n,r)=>n+(r.current[key]||0),0); const cards=[['Запусков',runs.length],['Снимков',sum('ticks')||runs.reduce((n,r)=>n+r.ticks.length,0)],['Событий',runs.reduce((n,r)=>n+r.events.reduce((a,e)=>a+e.count,0),0)],['Заявок',sum('generated')],['Завершено',sum('completed')],['Отказано',sum('rejected')]];
  $('cards').innerHTML=cards.map(([name,val])=>`<div class="card"><b>${{val}}</b><span>${{name}}</span></div>`).join('');
}}
function renderComparison(runs) {{
  if(!runs.length) {{ $('comparison').innerHTML='<div class="empty">Нет данных для выбранного фильтра.</div>'; return; }}
  const rows=runs.slice().sort((a,b)=>b.run_id-a.run_id).map(r=>`<tr><td>#${{r.run_id}}</td><td>${{esc(r.architecture)}}</td><td>${{esc(r.scenario)}}</td><td>${{esc(r.failure_mode)}}</td><td class="num">${{r.current.tick}}</td><td class="num">${{r.current.generated}}</td><td class="num">${{r.current.completed}}</td><td class="num">${{r.current.rejected}}</td><td class="num">${{r.current.mean_duration ?? '—'}}</td><td class="num">${{r.current.p95_duration ?? '—'}}</td><td class="num">${{r.current.completed_with_faults}}</td><td class="num">${{r.current.faulty}}</td><td class="num">${{r.current.success_rate}}%</td></tr>`).join('');
  $('comparison').innerHTML=`<div style="overflow:auto"><table><thead><tr><th>Run</th><th>Архитектура</th><th>Сценарий</th><th>Отказы</th><th class="num">Такт</th><th class="num">Заявки</th><th class="num">Готово</th><th class="num">Отказ</th><th class="num">Среднее T</th><th class="num">P95 T</th><th class="num">Готово через сбой</th><th class="num">Сбоев</th><th class="num">Успех</th></tr></thead><tbody>${{rows}}</tbody></table></div>`;
}}
function renderEvents(run) {{
  if(!run || !run.events.length) {{ $('events').innerHTML='<div class="empty">Нет событий.</div>'; return; }}
  const max=Math.max(...run.events.map(e=>e.count),1); $('events').innerHTML=run.events.map(e=>`<div class="barrow"><span class="barlabel">${{esc(e.event_type)}}</span><span class="bar"><i style="width:${{e.count/max*100}}%"></i></span><strong class="num">${{e.count}}</strong></div>`).join('');
}}
function renderChart(run) {{
  const svg=$('chart'); if(!run || !run.ticks.length) {{ svg.innerHTML=''; return; }} const W=800,H=280,L=42,R=14,T=20,B=32; const ticks=run.ticks; const values=[...ticks.flatMap(t=>[t.active_count||0,t.total_generated||0,t.completed_count||0,t.rejected_count||0])]; const max=Math.max(1,...values); const x=i=>L+(W-L-R)*(ticks.length<=1?0.5:i/(ticks.length-1)); const y=v=>(H-B)-(H-T-B)*v/max; const colors=['#4fc3f7','#ffca28','#66bb6a','#ef5350']; const names=['активно','сгенерировано','завершено','отказано']; let out=''; for(let g=0;g<=4;g++){{const gy=T+(H-T-B)*g/4;out+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{gy}}" y2="${{gy}}" stroke="#323842"/><text x="8" y="${{gy+4}}" fill="#90a4ae" font-size="11">${{Math.round(max*(4-g)/4)}}</text>`;}} names.forEach((name,s)=>{{const key=['active_count','total_generated','completed_count','rejected_count'][s];const path=ticks.map((t,i)=>`${{i?'L':'M'}} ${{x(i).toFixed(1)}} ${{y(t[key]||0).toFixed(1)}}`).join(' ');out+=`<path d="${{path}}" fill="none" stroke="${{colors[s]}}" stroke-width="2"/>`; out+=`<text x="${{L+s*130}}" y="14" fill="${{colors[s]}}" font-size="11">● ${{name}}</text>`;}}); out+=`<rect x="${{L}}" y="${{T}}" width="${{W-L-R}}" height="${{H-T-B}}" fill="none" stroke="#323842"/>`; svg.innerHTML=out;
}}
function renderHeatmap(run) {{ const el=$('heatmap'); if(!run || !run.neurons.length){{el.innerHTML='<div class="empty">Нет состояния нейронов.</div>';return;}} const maxRow=Math.max(...run.neurons.map(n=>n.row)),maxCol=Math.max(...run.neurons.map(n=>n.col)); const cls=n=>n.state_code==='FAULTY'?'fault':n.state_code==='BUSY'?'busy':n.state_code==='EXHAUSTED'?'done':'free'; el.innerHTML=`<div class="heat" style="grid-template-columns:repeat(${{maxCol+1}},minmax(38px,1fr))">${{run.neurons.map(n=>`<div class="cell ${{cls(n)}} ${{n.pending_fault?'pending':''}}"><span>${{n.row}},${{n.col}}</span><small>${{n.processed_requests_count||0}}/${{n.unprocessed_requests_count||0}}</small></div>`).join('')}}</div><p style="font-size:12px">Сбои: <b>${{run.current.faulty}}</b> · отложенные: <b>${{run.current.pending}}</b>. В ячейке: обработано / не принято.</p>`; }}
function render() {{ const runs=filteredRuns(); renderFilters(); const current=selected(); renderCards(runs); renderComparison(runs); renderEvents(current); renderChart(current); renderHeatmap(current); }}
$('architecture').addEventListener('change',()=>{{selectedRun=null;render();}}); $('scenario').addEventListener('change',()=>{{selectedRun=null;render();}}); $('run').addEventListener('change',()=>{{selectedRun=$('run').value;render();}}); $('reset').addEventListener('click',()=>{{ $('architecture').value=''; $('scenario').value=''; selectedRun=null; render(); }});
$('dbName').textContent=DATA.database.split('\\').pop(); $('generated').textContent=new Date(DATA.generated_at).toLocaleString('ru-RU'); render();
</script>
</body></html>'''


def generate_report(db_path: str | Path, output: str | Path | None = None) -> Path:
    db_path = Path(db_path)
    output_path = Path(output) if output else db_path.parent / "report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(load_report_data(db_path)), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained NS ADC analytics dashboard")
    parser.add_argument("--db", type=Path, default=Path(__file__).with_name("analytics") / "simulation.sqlite")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(generate_report(args.db, args.out))


if __name__ == "__main__":
    main()
