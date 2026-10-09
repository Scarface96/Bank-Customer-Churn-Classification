"""Run the full churn analysis and write the website to site/index.html.

    python -m analysis.build
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import data, model
from .report import BLUE, GRID, INK_2, MUTED, ORANGE, SERIES, Report, money, pct, style, to_json

REPO = "Scarface96/Bank-Customer-Churn-Classification"


# ------------------------------------------------------------------ figures --
def segment_figure(df: pd.DataFrame, overall: float) -> tuple[go.Figure, pd.DataFrame]:
    panels = [
        ("Country", "Geography", None),
        ("Age", "age_band", None),
        ("Products held", "products", ["1", "2", "3", "4"]),
        ("Activity", "IsActiveMember", None),
    ]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[p[0] for p in panels], shared_yaxes=True, horizontal_spacing=0.04)
    tables = []
    for i, (title, col, order) in enumerate(panels, 1):
        t = data.churn_rate(df, col)
        if col == "IsActiveMember":
            t[col] = t[col].map({0: "Inactive", 1: "Active"})
        t[col] = t[col].astype(str)
        if order:
            t = t.set_index(col).loc[order].reset_index()
        fig.add_bar(
            x=t[col], y=t["churn_rate"], marker_color=BLUE, marker_line_width=0,
            customdata=np.stack([t["customers"], t["churned"]], axis=1),
            hovertemplate="%{x}<br>Churn rate %{y:.1%}<br>%{customdata[1]:,} of %{customdata[0]:,} customers<extra></extra>",
            showlegend=False, row=1, col=i,
        )
        fig.add_hline(y=overall, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=i)
        tables.append(t.rename(columns={col: "group"}).assign(segment=title))
    style(fig, height=380)
    fig.update_annotations(font=dict(size=14, color=INK_2))
    fig.add_annotation(text=f"Bank average {overall:.0%}", x=1, xref="x4 domain", y=overall, yref="y4", showarrow=False, yshift=10, xanchor="right", font=dict(size=12, color=MUTED))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    table = pd.concat(tables)[["segment", "group", "customers", "churned", "churn_rate"]]
    table["churn_rate"] = (table["churn_rate"] * 100).round(1)
    return fig, table.rename(columns={"churn_rate": "churn rate %"})


def model_figure(comparison: pd.DataFrame, fitted: dict, s: model.Split) -> go.Figure:
    fig = go.Figure()
    order = list(model.MODELS)  # fixed colour per model, regardless of rank
    for i, name in enumerate(order):
        pts = model.roc_points(s.y_test, fitted[name][1])
        auc = comparison.set_index("model").loc[name, "test_roc_auc"]
        fig.add_scatter(
            x=pts["fpr"], y=pts["tpr"], mode="lines", name=f"{name} ({auc:.3f})",
            line=dict(color=SERIES[i], width=2),
            hovertemplate=f"{name}<br>False alarms %{{x:.0%}}<br>Churners caught %{{y:.0%}}<extra></extra>",
        )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dot", width=1), hoverinfo="skip", showlegend=False)
    style(fig, height=460)
    fig.update_xaxes(title="False alarms (share of loyal customers flagged)", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Churners caught", tickformat=".0%", range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="v", y=0.04, yanchor="bottom", x=0.98, xanchor="right", bgcolor="rgba(252,252,251,.9)"))
    return fig


def importance_figure(imp: pd.DataFrame) -> go.Figure:
    t = imp.sort_values("auc_drop")
    fig = go.Figure(
        go.Bar(
            x=t["auc_drop"], y=t["label"], orientation="h", marker_color=BLUE,
            error_x=dict(type="data", array=t["std"], color=MUTED, thickness=1, width=3),
            hovertemplate="%{y}<br>AUC drops by %{x:.3f} when shuffled<extra></extra>",
        )
    )
    style(fig, height=400)
    fig.update_xaxes(title="Drop in ROC AUC when this input is scrambled", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def lift_figure(lift: pd.DataFrame, overall: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[f"{d}" for d in lift["decile"]], y=lift["churn_rate"], marker_color=BLUE,
            customdata=np.stack([lift["churners"], lift["customers"], lift["share_of_all_churners"]], axis=1),
            hovertemplate="Risk group %{x}<br>Churn rate %{y:.0%}<br>%{customdata[0]} of %{customdata[1]} customers<br>Groups 1–%{x} hold %{customdata[2]:.0%} of all churners<extra></extra>",
        )
    )
    fig.add_hline(y=overall, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=f"Average {overall:.0%}", annotation_position="top right", annotation_font_color=MUTED)
    style(fig, height=380)
    fig.update_xaxes(title="Customers ranked by predicted risk, in tenths (1 = riskiest 10%)", type="category")
    fig.update_yaxes(title="Actual churn rate", tickformat=".0%")
    return fig


# --------------------------------------------------------- interactive HTML --
CALCULATOR = """
<form class="controls" id="calc" onsubmit="return false">
  <label>Country<select name="Geography"><option>France</option><option>Germany</option><option>Spain</option></select></label>
  <label>Gender<select name="Gender"><option>Female</option><option>Male</option></select></label>
  <label>Age<input name="Age" type="number" min="18" max="92" value="42"></label>
  <label>Products held<select name="NumOfProducts"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label>
  <label>Active member<select name="IsActiveMember"><option value="1">Yes</option><option value="0">No</option></select></label>
  <label>Balance<input name="Balance" type="number" min="0" step="1000" value="75000"></label>
  <label>Credit score<input name="CreditScore" type="number" min="350" max="850" value="650"></label>
  <label>Tenure (years)<input name="Tenure" type="number" min="0" max="10" value="5"></label>
  <label>Salary<input name="EstimatedSalary" type="number" min="0" step="1000" value="100000"></label>
  <label>Has credit card<select name="HasCrCard"><option value="1">Yes</option><option value="0">No</option></select></label>
</form>
<div class="readout" aria-live="polite">
  <div><b id="calc-p">–</b><span>chance this customer leaves</span></div>
  <div><b id="calc-band">–</b><span>risk band</span></div>
</div>
<p class="note" id="calc-why"></p>
"""

PLANNER = """
<form class="controls" id="plan" onsubmit="return false">
  <label>Value of keeping a customer: <output id="v-out"></output><input type="range" id="v" min="200" max="5000" step="100" value="1500"></label>
  <label>Cost of a retention offer: <output id="c-out"></output><input type="range" id="c" min="10" max="500" step="10" value="100"></label>
  <label>Offers that work: <output id="s-out"></output><input type="range" id="s" min="5" max="80" step="5" value="35"></label>
</form>
<div class="readout" aria-live="polite">
  <div><b id="p-best">–</b><span>best risk cut-off</span></div>
  <div><b id="p-contact">–</b><span>customers to contact</span></div>
  <div><b id="p-caught">–</b><span>churners reached</span></div>
  <div><b id="p-net">–</b><span>expected net value</span></div>
</div>
<figure class="chart"><div id="plan-chart" style="height:380px"></div></figure>
"""


def calculator_js(spec: dict, avg: float) -> str:
    return f"""
(function(){{
const M={to_json(spec)}, AVG={avg};
const LABEL={{CreditScore:'credit score',Tenure:'tenure',Balance:'balance',EstimatedSalary:'salary',HasCrCard:'has a credit card',IsActiveMember:'activity',Geography:'country',Gender:'gender',NumOfProducts:'number of products',Age:'age'}};
const f=document.getElementById('calc');
function score(){{
  const v=Object.fromEntries(new FormData(f).entries());
  const parts=[]; let z=M.intercept;
  const a=+v.Age, ax=[a,(a-45)**2/100];
  let ac=0; for(let i=0;i<2;i++) ac+=M.age.weight[i]*(ax[i]-M.age.mean[i])/M.age.scale[i];
  z+=ac; parts.push(['Age '+a,ac]);
  for(const [k,s] of Object.entries(M.numeric)){{const c=s.weight*((+v[k])-s.mean)/s.scale; z+=c; parts.push([LABEL[k],c]);}}
  for(const [k,w] of Object.entries(M.flags)){{const c=w*(+v[k]); z+=c; if(k==='IsActiveMember') parts.push([+v[k]?'Active member':'Inactive member', c-w*0.5]);}}
  for(const [k,t] of Object.entries(M.categories)){{const c=t[v[k]]||0; z+=c; const label=k==='NumOfProducts'?v[k]+' product'+(v[k]==='1'?'':'s'):v[k]; parts.push([label,c]);}}
  const p=1/(1+Math.exp(-z));
  document.getElementById('calc-p').textContent=(p*100).toFixed(0)+'%';
  document.getElementById('calc-band').textContent=p>=0.5?'High':p>=0.25?'Elevated':p>=AVG*0.6?'Typical':'Low';
  const up=parts.filter(x=>x[1]>0.15).sort((a,b)=>b[1]-a[1]).slice(0,3).map(x=>x[0]);
  const down=parts.filter(x=>x[1]<-0.15).sort((a,b)=>a[1]-b[1]).slice(0,3).map(x=>x[0]);
  document.getElementById('calc-why').textContent=(up.length?'Pushing risk up: '+up.join(', ')+'. ':'')+(down.length?'Holding it down: '+down.join(', ')+'.':'');
}}
f.addEventListener('input',score); score();
}})();
"""


def planner_js(y: np.ndarray, p: np.ndarray) -> str:
    pairs = [[round(float(a), 4), int(b)] for a, b in zip(p, y)]
    return f"""
(function(){{
const D={to_json(pairs)};
const T=Array.from({{length:96}},(_,i)=>0.03+i*0.01);
const $=id=>document.getElementById(id);
const fmt=v=>(v<0?'−':'')+'$'+Math.abs(Math.round(v)).toLocaleString();
function run(){{
  const val=+$('v').value, cost=+$('c').value, ok=+$('s').value/100;
  $('v-out').textContent='$'+val.toLocaleString(); $('c-out').textContent='$'+cost; $('s-out').textContent=$('s').value+'%';
  let best=null; const net=[], hover=[];
  for(const t of T){{
    let n=0,c=0; for(const [p,y] of D) if(p>=t){{n++; if(y) c++;}}
    const v=c*ok*val-n*cost; net.push(v); hover.push([n,c]);
    if(!best||v>best.v) best={{t,v,n,c}};
  }}
  $('p-best').textContent=(best.t*100).toFixed(0)+'%+';
  $('p-contact').textContent=best.n.toLocaleString();
  $('p-caught').textContent=best.c.toLocaleString();
  $('p-net').textContent=fmt(best.v);
  Plotly.react('plan-chart',[
    {{x:T,y:net,customdata:hover,type:'scatter',mode:'lines',line:{{color:'{BLUE}',width:2}},name:'Net value',
      hovertemplate:'Contact everyone at %{{x:.0%}} risk or more<br>%{{customdata[0]}} contacted, %{{customdata[1]}} churners reached<br>Net value %{{y:$,.0f}}<extra></extra>'}},
    {{x:[best.t],y:[best.v],type:'scatter',mode:'markers',marker:{{color:'{ORANGE}',size:11,line:{{color:'#fcfcfb',width:2}}}},name:'Best cut-off',hoverinfo:'skip'}}
  ],{{height:380,margin:{{l:8,r:16,t:16,b:8}},paper_bgcolor:'#fcfcfb',plot_bgcolor:'#fcfcfb',showlegend:false,
     font:{{family:'"Public Sans",system-ui,sans-serif',size:13,color:'{INK_2}'}},hoverlabel:{{bgcolor:'white',bordercolor:'#c3c2b7'}},
     xaxis:{{title:{{text:'Contact customers whose predicted risk is at least…'}},tickformat:'.0%',showgrid:false,linecolor:'#c3c2b7',automargin:true}},
     yaxis:{{title:{{text:'Expected net value, 2,000 held-out customers'}},tickprefix:'$',gridcolor:'{GRID}',zeroline:true,zerolinecolor:'#c3c2b7',automargin:true}}}},
    {{displaylogo:false,responsive:true}});
}}
['v','c','s'].forEach(id=>$(id).addEventListener('input',run)); run();
}})();
"""


# --------------------------------------------------------------------- main --
def main(out="site/index.html"):
    raw = data.load()
    df = data.add_features(raw)
    overall = df[data.TARGET].mean()

    s = model.split(df)
    comparison, fitted = model.compare(s)
    best_name = comparison.loc[0, "model"]
    best_model, best_p = fitted[best_name]
    simple_model, simple_p = fitted["Explainable logistic regression"]
    imp = model.importance(best_model, s)
    lift = data.lift_table(s.y_test, best_p)
    top2 = lift.loc[lift["decile"] == 2, "share_of_all_churners"].iloc[0]
    spec = model.export_logistic(simple_model)

    seg_fig, seg_table = segment_figure(df, overall)
    by = lambda col, val: df.loc[df[col] == val, data.TARGET].mean()
    germany, inactive, three_plus = by("Geography", "Germany"), by("IsActiveMember", 0), df.loc[df["NumOfProducts"] >= 3, data.TARGET].mean()
    two = by("NumOfProducts", 2)
    base_auc = comparison.set_index("model").loc["Logistic regression (original)", "test_roc_auc"]
    simple_auc = comparison.set_index("model").loc["Explainable logistic regression", "test_roc_auc"]
    best_auc = comparison.loc[0, "test_roc_auc"]

    r = Report(
        title=f"The riskiest fifth of customers accounts for {top2:.0%} of those who leave",
        project="Bank Customer Churn",
        summary=(
            f"Across 10,000 customers, {overall:.1%} closed their accounts. Germany, inactive members and anyone holding "
            "three or more products leave far more often. A model trained on these patterns ranks customers well enough "
            "that a targeted retention campaign pays for itself."
        ),
        repo=REPO,
        accent=BLUE,
        source="Bank_Churn.csv: 10,000 bank customers in France, Germany and Spain, with demographics, balances, product holdings and whether they left.",
        method=(
            "pandas for preparation; scikit-learn pipelines for logistic regression, a tuned random forest and gradient boosting, "
            "compared with 5-fold stratified cross-validation on 80% of customers and checked once on the held-out 20%. "
            "Driver strength comes from permutation importance. The calculator and planner run in your browser on numbers exported by the build."
        ),
    )
    r.kpis([
        ("10,000", "customers analysed"),
        (pct(overall), "churned"),
        (f"{best_auc:.2f}", "ROC AUC of the best model", f"{best_name.lower()}, held-out data"),
        (f"{top2:.0%}", "of churners in the riskiest 20%"),
    ])

    r.section(
        "Who leaves the bank?",
        f"<p>Churn is concentrated. <b>German customers churn at {germany:.0%}</b>, about twice the rate in France or Spain. "
        f"<b>Inactive members leave at {inactive:.0%}</b>. The sharpest signal is product count: customers with two products "
        f"churn at only {two:.0%}, but <b>{three_plus:.0%} of those with three or four products leave</b>. Risk also climbs steadily with age into the fifties.</p>",
        fig=seg_fig, table=seg_table,
        note="Dotted line: the bank-wide churn rate. Hover a bar for the customer count.",
    )

    cmp_table = comparison.copy()
    for c in ["cv_roc_auc", "cv_std", "test_roc_auc", "test_pr_auc", "test_accuracy"]:
        cmp_table[c] = cmp_table[c].round(3)
    r.section(
        "Which model should the bank trust?",
        f"<p>The original notebook's logistic regression reached an ROC AUC of {base_auc:.2f}. Two small changes, "
        f"letting age bend (risk peaks in middle age) and treating product count as a category, lift the same transparent model "
        f"to <b>{simple_auc:.2f}</b>. The tree models edge ahead at <b>{best_auc:.3f}</b>, because they find interactions on their own.</p>"
        "<p>Each curve shows the trade-off: moving right flags more loyal customers by mistake, moving up catches more real churners.</p>",
        fig=model_figure(comparison, fitted, s),
        table=cmp_table.rename(columns={"cv_roc_auc": "CV ROC AUC", "cv_std": "CV std", "test_roc_auc": "test ROC AUC", "test_pr_auc": "test PR AUC", "test_accuracy": "test accuracy"}),
        table_caption="Show the scores",
    )

    r.section(
        "What drives the prediction?",
        f"<p>Scramble one input at a time and see how much the {best_name.lower()} gets worse. <b>{imp.loc[0, 'label']}</b> and "
        f"<b>{imp.loc[1, 'label'].lower()}</b> carry most of the signal, followed by activity and country. Salary, card ownership "
        "and tenure barely matter, so they're poor levers for a retention strategy.</p>",
        fig=importance_figure(imp),
        table=imp[["label", "auc_drop", "std"]].round(4).rename(columns={"label": "input", "auc_drop": "AUC drop", "std": "spread"}),
        note="Bars show the average drop over 8 shuffles; whiskers show the spread.",
    )

    lift_t = lift.copy()
    lift_t["churn_rate"] = (lift_t["churn_rate"] * 100).round(1)
    lift_t["share_of_all_churners"] = (lift_t["share_of_all_churners"] * 100).round(1)
    lift_t["avg_risk"] = (lift_t["avg_risk"] * 100).round(1)
    r.section(
        "Where should retention effort go?",
        f"<p>Rank the held-out customers by predicted risk and split them into tenths. The riskiest tenth churns at "
        f"<b>{lift.loc[0, 'churn_rate']:.0%}</b>, against {overall:.0%} on average, and the riskiest fifth contains "
        f"<b>{top2:.0%} of everyone who actually left</b>. Calling those customers first is several times more efficient than calling at random.</p>",
        fig=lift_figure(lift, overall),
        table=lift_t.rename(columns={"decile": "risk group", "avg_risk": "avg predicted risk %", "churn_rate": "churn rate %", "share_of_all_churners": "cumulative share of churners %"}),
    )

    r.section(
        "Score a customer",
        "<p>Change the details to see how the explainable model rates a customer. It's the transparent logistic regression "
        f"(ROC AUC {simple_auc:.2f}), so every factor's push is visible.</p>",
        html=CALCULATOR,
        note="For illustration only. A real decision would use the bank's own, more recent data.",
    )
    r.script(calculator_js(spec, overall))

    r.section(
        "How many customers should the bank call?",
        "<p>A retention offer costs money, and not every churner can be saved. Set what a kept customer is worth, what an offer "
        "costs and how often offers work. The chart finds the risk cut-off that earns the most, using the best model's "
        "predictions for the 2,000 held-out customers.</p>",
        html=PLANNER,
        note="Contacting nearly everyone wastes offers on customers who would have stayed; contacting too few misses savable churners. Raise the offer cost or lower the success rate and the best cut-off moves up.",
    )
    r.script(planner_js(s.y_test.to_numpy(), best_p))

    path = r.write(out)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB). Best model: {best_name} (test ROC AUC {best_auc:.3f}).")
    return r


if __name__ == "__main__":
    main()
