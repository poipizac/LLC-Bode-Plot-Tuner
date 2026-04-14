import numpy as np
import plotly.graph_objects as go
from scipy import signal, optimize
from fasthtml.common import *
import json

# FastHTML App Setup
app, rt = fast_app(hdrs=(
    picolink,
    Script(src="https://cdn.plot.ly/plotly-2.32.0.min.js"),
    Script(src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"),
    Script("mermaid.initialize({startOnLoad:true, theme:'dark', securityLevel:'loose'});"),
    Style("""
        * { box-sizing: border-box; }
        body { background-color: #0f172a; color: #f1f5f9; font-family: 'Inter', sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden; }
        .sidebar { width: 380px; height: 100vh; background: #1e293b; padding: 20px; position: fixed; border-right: 1px solid #334155; overflow-y: auto; box-sizing: border-box; z-index: 100; }
        .main-content { margin-left: 380px; flex-grow: 1; padding: 40px; background: #0f172a; height: 100vh; overflow-y: auto; box-sizing: border-box; }
        .dashboard-grid { 
            display: grid; 
            grid-template-columns: minmax(0, 1fr); 
            grid-auto-rows: 400px;
            grid-row-gap: 50px; 
            grid-column-gap: 30px; 
            margin-top: 35px; 
        }
        .chart-container { 
            background: #1e293b; border: 1px solid #334155; border-radius: 12px; 
            position: relative; width: 100%; height: 100%; overflow: hidden; box-shadow: 0 10px 15px rgba(0,0,0,0.3);
        }
        .chart-header { 
            position: absolute; top: 0; left: 0; right: 0; height: 48px;
            display: flex; align-items: center; justify-content: center; 
            background: #1e293b; border-bottom: 1px solid #334155; 
            font-weight: 700; color: #94a3b8; text-transform: uppercase; font-size: 0.9em; letter-spacing: 0.05em;
            z-index: 10;
        }
        .chart-div { 
            position: absolute; top: 48px; left: 0; right: 0; bottom: 0; padding: 0; 
        }
        .unit-label-y { position: absolute; top: 60px; left: 15px; z-index: 20; color: #94a3b8; font-size: 12px; font-weight: 700; background: rgba(15,23,42,0.8); padding: 2px 6px; border-radius: 4px; }
        .unit-label-x { position: absolute; bottom: 5px; right: 15px; z-index: 20; color: #94a3b8; font-size: 12px; font-weight: 700; background: rgba(15,23,42,0.8); padding: 2px 6px; border-radius: 4px; }
        .custom-legend { position: absolute; top: 60px; right: 15px; display: flex; gap: 15px; z-index: 20; background: rgba(30,41,59,0.8); padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #f1f5f9; font-weight: 600; }
        .metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); grid-gap: 20px; margin-bottom: 20px; position: relative; }
        .metric-card { background: #1e293b; border: 1px solid #334155; padding: 18px; border-radius: 12px; text-align: center; }
        .metric-label { font-size: 0.8em; color: #94a3b8; text-transform: uppercase; }
        .metric-value { font-size: 1.6em; font-weight: 700; color: #3b82f6; }
        .sync-spinner { display: none; position: absolute; right: -40px; top: 35%; color: #3b82f6; font-size: 0.8em; font-weight: 600; }
        .htmx-request .sync-spinner { display: block; animation: blink 1s infinite; }
        @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        .control-group { margin-bottom: 12px; padding: 10px; background: #0f172a; border-radius: 8px; border: 1px solid #334155; }
        .control-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .num-input { width: 95px !important; background: #1e293b !important; color: #3b82f6 !important; border: 1px solid #4b5563 !important; border-radius: 6px; text-align: right; padding: 2px 8px !important; font-size: 0.85em !important; font-family: monospace; }
        input[type="range"] { width: 100%; accent-color: #3b82f6; cursor: pointer; }
        .schematic-box { margin-top: 15px; background: #2c2c2c; padding: 10px; border-radius: 10px; border: 1px solid #4b5563; height: 180px; overflow: hidden; }
        .hw-tag { margin-top: 15px; font-size: 0.7em; color: #10b981; font-weight: 700; text-align: center; border: 1px solid #10b981; border-radius: 4px; padding: 4px; }
        h2 { margin: 0 0 10px 0; font-size: 1.5em; font-weight: 800; }
    """)
))

# PHYSICS CONSTANTS
Lr, Cr, Lm = 90e-6, 47e-9, 450e-6
Vin, Vout, Pout_max = 395, 20, 200
# FIXED Hardware CALIBRATION factor (targets 1.462 kHz)
CALIBRATION_GAIN_V_KHZ = -1130.0 
R_upper, R_led, CTR, R_pullup = 69800, 2000, 10.0, 5000 

def get_snub_phase(f, r65, c73):
    """Opto snubber lead-lag phase contribution."""
    if c73 <= 0: return np.zeros_like(f)
    fz_snub = 1.0 / (2 * np.pi * r65 * c73)
    fp_snub = 1.0 / (2 * np.pi * (r65 + 20.0) * c73)  # 20Ω LED dynamic impedance
    return np.arctan(f / fz_snub) * (180.0/np.pi) - np.arctan(f / fp_snub) * (180.0/np.pi)

def get_snub_mag(f, r65, c73):
    """Opto snubber lead-lag magnitude contribution (dB)."""
    if c73 <= 0: return np.zeros_like(f)
    fz_snub = 1.0 / (2 * np.pi * r65 * c73)
    fp_snub = 1.0 / (2 * np.pi * (r65 + 20.0) * c73)
    return 10 * np.log10(1 + (f / fz_snub)**2) - 10 * np.log10(1 + (f / fp_snub)**2)

def get_lc_filter(f, lf51, cout2, Rload, enable):
    """Post LC filter 2nd-order magnitude (dB) and phase (deg)."""
    if not enable or lf51 <= 0 or cout2 <= 0:
        return np.zeros_like(f), np.zeros_like(f)
    f0 = 1.0 / (2 * np.pi * np.sqrt(lf51 * cout2))
    Q = Rload * np.sqrt(cout2 / lf51)
    u = f / f0
    mag = -10 * np.log10((1 - u*u)**2 + (u/Q)**2)
    ph = np.arctan2(-u / Q, 1 - u*u) * (180.0/np.pi)
    ph = np.where(ph > 0, ph - 360.0, ph)  # keep phase continuously falling
    return mag, ph

def get_total_phase(f, Rload, cout, resr, r68, c71, c70, Td, GBW, r65=51, c73=0, lf51=0, cout2=0, enable_lc=False, fp_llc=3500, Q_llc=1.0):
    """Calculates unwrapped monotonically decreasing phase lag analytically."""
    f = np.maximum(f, 1e-12)
    # Plant (LLC Double Pole + ESR Zero)
    w_p_llc = 2 * np.pi * fp_llc
    fz_esr = 1.0 / (2 * np.pi * resr * cout) if resr > 1e-12 else 1e15
    u = f / fp_llc
    phi_dp = np.arctan2(-u / Q_llc, 1 - u*u) * (180.0/np.pi)
    phi_dp = np.where(phi_dp > 0, phi_dp - 360.0, phi_dp)  # continuously falling
    phi_p = phi_dp + np.arctan(f/fz_esr) * (180.0/np.pi)
    
    # Compensator (Type II)
    fz_comp = 1.0 / (2 * np.pi * r68 * c71) if c71 > 0 else 1e15
    fp_comp = 1.0 / (2 * np.pi * r68 * c70) if c70 > 0 else 1e15
    phi_c = -90.0 + np.arctan(f/fz_comp) * (180.0/np.pi)
    if c70 > 0: phi_c -= np.arctan(f/fp_comp) * (180.0/np.pi)
        
    # Parasitics
    phi_delay = -360.0 * f * (Td * 1e-6)
    phi_opamp = -np.arctan(f / (GBW * 1e6)) * (180.0/np.pi)
    
    # Opto Snubber
    phi_snub = get_snub_phase(f, r65, c73)
    
    # Post LC Filter
    _, phi_lc = get_lc_filter(f, lf51, cout2, Rload, enable_lc)
    
    return phi_p + phi_c + phi_delay + phi_opamp + phi_snub + phi_lc, phi_p, phi_c

def find_crossover(loop_tf, Rload, cout, resr, r68, c71, c70, Td, GBW, gain_trim, r65=51, c73=0, lf51=0, cout2=0, enable_lc=False, fp_llc=3500, Q_llc=1.0):
    def mag_err(logf):
        f_val = np.power(10, logf); w = 2 * np.pi * f_val
        _, mag, _ = signal.bode(loop_tf, [w])
        snub_m = get_snub_mag(np.array([f_val]), r65, c73)[0]
        lc_m, _ = get_lc_filter(np.array([f_val]), lf51, cout2, Rload, enable_lc)
        return mag[0] + gain_trim + snub_m + lc_m[0]
    try:
        logf_fc = optimize.brentq(mag_err, -1, 7) 
        fc = np.power(10, logf_fc)
        ph_math, _, _ = get_total_phase(fc, Rload, cout, resr, r68, c71, c70, Td, GBW, r65, c73, lf51, cout2, enable_lc, fp_llc, Q_llc)
        return fc, 180 + ph_math
    except: return None, None

def find_phase_crossover(Rload, cout, resr, r68, c71, c70, Td, GBW, loop_tf, gain_trim, r65=51, c73=0, lf51=0, cout2=0, enable_lc=False, fp_llc=3500, Q_llc=1.0):
    def ph_err(logf):
        f_val = np.power(10, logf)
        ph, _, _ = get_total_phase(f_val, Rload, cout, resr, r68, c71, c70, Td, GBW, r65, c73, lf51, cout2, enable_lc, fp_llc, Q_llc)
        return ph + 180.0
    try:
        logf_pc = optimize.brentq(ph_err, -1, 7) 
        fc_ph = np.power(10, logf_pc)
        _, mag_pc, _ = signal.bode(loop_tf, [2 * np.pi * fc_ph])
        snub_m = get_snub_mag(np.array([fc_ph]), r65, c73)[0]
        lc_m, _ = get_lc_filter(np.array([fc_ph]), lf51, cout2, Rload, enable_lc)
        return -(mag_pc[0] + gain_trim + snub_m + lc_m[0])
    except: return float('inf')

def get_base_layout(x_range, y_range, log=True, y_dtick=None):
    lyt = dict(
        showlegend=False,
        margin=dict(l=60, r=30, t=30, b=50, autoexpand=False), # Explicitly disable autoexpand
        height=340, 
        autosize=False,
        uirevision='constant',  
        transition=dict(duration=0), # explicitly block tween sequences
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        hoverlabel=dict(bgcolor="rgba(15,23,42,0.9)", font_size=12, font_family="Inter", bordercolor="#334155"),
        xaxis=dict(
            type='log' if log else 'linear', gridcolor='#1e293b', automargin=False, showgrid=True,
            showspikes=True, spikethickness=2, spikedash="dash", spikecolor="#a855f7"
        ),
        yaxis=dict(gridcolor='#1e293b', automargin=False, showgrid=True)
    )
    if log:
        lyt['xaxis']['tickvals'] = [1, 10, 100, 1000, 10000, 100000, 1000000]
        lyt['xaxis']['ticktext'] = ["1", "10", "100", "1k", "10k", "100k", "1M"]
        lyt['xaxis']['minor'] = dict(ticks="inside", ticklen=4, showgrid=False)
        
    if x_range is not None: lyt['xaxis']['range'] = x_range
    if y_range is not None: 
        lyt['yaxis']['range'] = y_range
        lyt['yaxis']['tick0'] = y_range[0]
        
    if y_dtick is not None:
        lyt['yaxis']['dtick'] = y_dtick
        
    return lyt

def generate_analysis(Lm, Lr, Cr, n, Td, GBW, r68, c71, c70, cout, resr, load_pct, fp_llc=3500, Q_llc=1.0, phase_view="math", gain_trim=-40.0, vout=20, pout=200, r65=51, c73=0, enable_lc=False, lf51=4e-6, cout2=0.1e-6):
    load_pct_safe = max(0.001, load_pct)
    Rload = (vout * vout) / (pout * load_pct_safe)
    
    # LLC Double Pole Plant Model
    G_dc_vcap = 42.0 * (n / 9.66) 
    
    w_z1 = 1.0 / (resr * cout) if resr > 1e-6 else 1e12
    w_p_llc = 2 * np.pi * fp_llc
    
    tf_plant = signal.TransferFunction(
        [G_dc_vcap / w_z1, G_dc_vcap],
        [(1.0/w_p_llc)**2, 1.0/(w_p_llc*Q_llc), 1.0]
    )
    
    # Compensator Gc(s) + Op-Amp GBW
    eps = 1e-8
    K_stat = (CTR * R_pullup) / (R_upper * R_led * c71) if c71 > 0 else 1.0
    num_c = [r68 * c71, 1]
    den_c = [r68 * c70, 1, eps] if c70 > 1e-12 else [1, eps]
    
    w_gbw_pole = 2 * np.pi * (GBW * 1e6)
    tf_comp = signal.TransferFunction(
        np.polymul(np.array(num_c) * K_stat * -1.0, [1]), 
        np.polymul(den_c, [1/w_gbw_pole, 1])
    )
    
    loop_tf = signal.TransferFunction(np.polymul(tf_plant.num, tf_comp.num), np.polymul(tf_plant.den, tf_comp.den))
    
    w = np.logspace(0, 7, 1000) * 2 * np.pi; freq = w / (2 * np.pi)
    _, mag_p, _ = signal.bode(tf_plant, w)
    _, mag_c, _ = signal.bode(tf_comp, w)
    _, mag_l_raw, _ = signal.bode(loop_tf, w)
    
    # Opto Snubber & Post LC Filter magnitude contributions
    mag_snub = get_snub_mag(freq, r65, c73)
    mag_lc, _ = get_lc_filter(freq, lf51, cout2, Rload, enable_lc)
    
    # Apply Manual Gain Trim + additional TF contributions
    mag_extra = mag_snub + mag_lc
    mag_l = mag_l_raw + gain_trim + mag_extra
    # Shift Plant Trace for visual consistency in the Bode plot sum
    mag_p = mag_p + gain_trim
    
    # Analytical Phase Summation (Unwrapped) — includes snubber & LC
    phase_l_math, phase_p_math, phase_c_math = get_total_phase(freq, Rload, cout, resr, r68, c71, c70, Td, GBW, r65, c73, lf51, cout2, enable_lc, fp_llc, Q_llc)
    
    phase_p = phase_p_math
    phase_c = phase_c_math
    phase_l = phase_l_math
    
    # Stability Margins (with snubber & LC)
    fc, pm = find_crossover(loop_tf, Rload, cout, resr, r68, c71, c70, Td, GBW, gain_trim, r65, c73, lf51, cout2, enable_lc, fp_llc, Q_llc)
    gm = find_phase_crossover(Rload, cout, resr, r68, c71, c70, Td, GBW, loop_tf, gain_trim, r65, c73, lf51, cout2, enable_lc, fp_llc, Q_llc)
    
    h_cl = signal.TransferFunction(loop_tf.num, np.polyadd(loop_tf.den, loop_tf.num))
    t, y = signal.step(h_cl, T=np.linspace(0, 0.05, 1000))
    y_final = y[-1] if len(y) > 0 else 0
    settling_time = 0
    if y_final > 1e-6:
        for i in range(len(y)-1, 0, -1):
            if abs(y[i] - y_final) > 0.02 * y_final: settling_time = t[i]; break

    def create_fig(x_default, data_list, title, x_range, y_range, log=True, y_dtick=None):
        fig = go.Figure(layout=get_base_layout(x_range, y_range, log, y_dtick))
        for item in data_list:
            if len(item) == 4:
                custom_x, d, n, style = item
            else:
                d, n, style = item
                custom_x = x_default
                
            if style.get('mode') == 'markers':
                fig.add_trace(go.Scatter(x=custom_x, y=d, name=n, mode='markers', marker=style.get('marker'), uid=n, hoverinfo='skip'))
            else:
                fig.add_trace(go.Scatter(x=custom_x, y=d, name=n, line=style, uid=n, hovertemplate="%{y:.2f}"))
                
        if "Magnitude" in title:
            fig.add_hline(y=0, line_dash="solid", line_width=2, line_color="#000000")
        if "Phase" in title:
            fig.add_hline(y=(0 if phase_view == 'instrument' else -180), line_dash="dash", line_width=1.5, line_color="#ef4444")

        return fig.to_json()

    mag_traces = [
        ([1462.0], [0.0], "1.462kHz Target", dict(mode='markers', marker=dict(symbol='cross', size=14, color='#a855f7', line=dict(width=2, color='#a855f7')))),
        (mag_p, "Plant G<sub>vf</sub>", dict(color='#94a3b8', width=1.5, dash='dash')),
        (mag_c, "Compensator G<sub>c</sub>", dict(color='#fbbf24', width=1.5, dash='dot')),
        (mag_l, "Open-Loop T(s)", dict(color='#3b82f6', width=2.5))
    ]
                  
    ph_shift = 180.0 if phase_view == 'instrument' else 0.0
    ph_target_y = 128.8 if phase_view == 'instrument' else -51.2
    
    ph_traces = [
        ([1462.0], [ph_target_y], "1.462kHz Target", dict(mode='markers', marker=dict(symbol='cross', size=14, color='#a855f7', line=dict(width=2, color='#a855f7')))),
        (phase_p + ph_shift, "Plant G<sub>vf</sub>", dict(color='#94a3b8', width=1.5, dash='dash')),
        (phase_c + ph_shift, "Compensator G<sub>c</sub>", dict(color='#fbbf24', width=1.5, dash='dot')),
        (phase_l + ph_shift, "Open-Loop T(s)", dict(color='#3b82f6', width=2.5))
    ]
    
    ph_y_range = [-20, 200] if phase_view == 'instrument' else [-200, 20]
    return [create_fig(freq, mag_traces, "Loop Magnitude", [0, 6], [-60, 60], True, 20),
            create_fig(freq, ph_traces, "Loop Phase", [0, 6], ph_y_range, True, 50),
            create_fig(t, [(y, "Vout", dict(color='#10b981', width=3))], "Step Response", [0, 0.05], [0, 1.5], False, 0.5),
            create_fig(None, [], "Pole-Zero Map", None, None, False, None), # PZ x_range y_range set to None
            fc, pm, gm, h_cl, r68, c71, c70]

def HybridControl(label, id, val, min_v, max_v, step):
    return Div(cls="control-group")(
        Div(cls="control-header")(
            Label(label, cls="control-label"),
            Input(type="number", id=f"{id}-num", name=id, value=val, min=min_v, max=max_v, step=step, cls="num-input",
                  oninput=f"document.getElementById('{id}-slider').value = this.value;")
        ),
        Input(type="range", id=f"{id}-slider", value=val, min=min_v, max=max_v, step=step,
              oninput=f"document.getElementById('{id}-num').value = this.value;")
    )

def make_html_chart(html_id, title, y_unit, x_unit, legends=[]):
    leg_divs = []
    for n, c, s in legends:
        if s == 'cross': line_css = f"color: {c}; font-size: 14px; font-weight: bold;" 
        elif s == 'circle': line_css = f"color: {c}; font-size: 16px; font-weight: bold;" 
        else: line_css = f"background-color: {c}; width: 14px; height: 3px;" if s == 'solid' else f"border-bottom: 2px {s} {c}; width: 14px; height: 0;"
        marker = Div("×", style=line_css) if s == 'cross' else Div("○", style=line_css) if s == 'circle' else Div(style=line_css)
        leg_divs.append(Div(cls="legend-item")(marker, Div(n)))

    return Div(cls="chart-container")(
        Div(title, cls="chart-header"),
        Div(y_unit, cls="unit-label-y") if y_unit else "",
        Div(x_unit, cls="unit-label-x") if x_unit else "",
        Div(cls="custom-legend")(*leg_divs) if legends else "",
        Div(id=html_id, cls="chart-div")
    )

@rt("/")
def get():
    # DOM Content Load persistence logic

    persistence_script = Script("""
    document.addEventListener("DOMContentLoaded", () => {
        // Purge deprecated keys from previous versions
        ['Lm', 'Lr', 'Cr', 'r118', 'c107', 'c108'].forEach(k => localStorage.removeItem(k));

        const baseFields = ['phase_view', 'n', 'vout', 'pout', 'gain_trim', 'Td', 'GBW', 'r68', 'c71', 'c70', 'cout', 'resr', 'load_pct', 'fp_llc', 'Q_llc', 'r65', 'c73', 'enable_lc', 'lf51', 'cout2'];
        
        // 1. F5 記憶讀取
        baseFields.forEach(f => {
            let val = localStorage.getItem(f);
            if (val !== null) {
                document.querySelectorAll(`[name='${f}'], #${f}, [name='${f}-slider']`).forEach(el => {
                    if (el.type === 'checkbox') { el.checked = (val === 'true' || val === 'on'); }
                    else { el.value = val; }
                });
            }
        });

        // 2. 初始強制畫圖
        setTimeout(() => {
            let form = document.querySelector('form');
            if(form) htmx.trigger(form, 'change'); 
        }, 150);

        // 3. 數值同步與存檔
        document.body.addEventListener('input', (e) => {
            let key = e.target.name || e.target.id;
            if (!key) return;
            let baseKey = key.replace('-slider', '');
            if (baseFields.includes(baseKey)) {
                let val = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
                localStorage.setItem(baseKey, val);
                document.querySelectorAll(`[name='${baseKey}'], #${baseKey}, [name='${baseKey}-slider']`).forEach(el => { 
                    if (el !== e.target) { if (el.type === 'checkbox') el.checked = val; else el.value = val; }
                });
            }
        });
        
    });
    """)
    
    sch = "graph TD\n V1[V+] --- Co[Cout] --- Es[ESR] --- V2[V-]\n V1 --- Ru[R_up] --- Re[Ref] --- C1[C1] --- R2[R2] --- Ca[Ca]\n Re --- C2[C2] --- Ca\n style R2 fill:#0f172a,stroke:#00d1ff\n style C1 fill:#0f172a,stroke:#00d1ff"
    sidebar = Div(cls="sidebar")(
        H2("NXP TEA19161T Vcap LLC Loop Tuner", style="font-size:1.1em;"),
        Form(hx_post="/update", hx_target="#main-dashboard", hx_trigger="change delay:150ms, input delay:150ms")(
            Div(cls="control-group")(
                Label("Phase Display Mode", cls="control-label"),
                Select(
                    Option("Math (- Deg)", value="math"),
                    Option("Instrument (+ Deg)", value="instrument"),
                    name="phase_view", id="phase_view", cls="num-input", style="width: 100% !important; margin-top: 5px; text-align: center; color: #fff !important; font-weight: 600;"
                )
            ),
            H2("Power Stage", style="font-size:0.9em; margin-top:20px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;"),
            HybridControl("Transformer Ratio (n)", "n", 9.66, 1, 30, 0.01),
            HybridControl("Vout (V)", "vout", 20, 3.3, 60, 0.1),
            HybridControl("Pout (W)", "pout", 150, 10, 1000, 5),
            HybridControl("Load Level (%)", "load_pct", 50, 5, 100, 5),
            HybridControl("Cout Cap (µF)", "cout", 1690, 100, 5000, 10),
            HybridControl("ESR Res (mΩ)", "resr", 20, 1, 200, 1),
            H2("Compensator & Parasitics", style="font-size:0.9em; margin-top:20px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;"),
            HybridControl("Loop Gain Trim (dB)", "gain_trim", -40, -60, 60, 1),
            HybridControl("Propagation Delay (μs)", "Td", 0.5, 0, 10, 0.1),
            HybridControl("Op-Amp GBW (MHz)", "GBW", 2.0, 0.1, 20, 0.1),
            HybridControl("R68 Resistor (Ω)", "r68", 20000, 100, 1000000, 100),
            HybridControl("C71 Zero Cap (nF)", "c71", 100, 0.1, 1000, 0.1),
            HybridControl("C70 Pole Cap (nF)", "c70", 0.2, 0, 100, 0.01),
            H2("LLC Dynamics", style="font-size:0.9em; margin-top:20px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;"),
            HybridControl("LLC Double Pole (Hz)", "fp_llc", 3500, 2000, 5000, 10),
            HybridControl("LLC DP Q-factor", "Q_llc", 1.0, 0.1, 3.0, 0.1),
            H2("Opto Snubber", style="font-size:0.9em; margin-top:20px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;"),
            HybridControl("R65 (Ω)", "r65", 51, 10, 5000, 1),
            HybridControl("C73 (nF)", "c73", 10, 0, 100, 0.1),
            H2("Post LC Filter", style="font-size:0.9em; margin-top:20px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;"),
            Div(cls="control-group")(
                Div(cls="control-header")(
                    Label("Enable LF51", cls="control-label"),
                    Input(type="checkbox", name="enable_lc", id="enable_lc", style="accent-color:#3b82f6; width:20px; height:20px; cursor:pointer;")
                )
            ),
            HybridControl("LF51 (μH)", "lf51", 4.0, 0.1, 100, 0.1),
            HybridControl("Cout2 (μF)", "cout2", 0.1, 0.01, 10, 0.01),
        ),
        Div(cls="hw-tag")("Calibrated to Hardware Ver 1.0 (AP4310)"),
        Div(cls="schematic-box")(Div(sch, cls="mermaid")),
    )
    leg_mag = [("1.462kHz Target", "#a855f7", "cross"), ("Plant Gvf", "#94a3b8", "dashed"), ("Compensator Gc", "#fbbf24", "dotted"), ("Open-Loop T(s)", "#3b82f6", "solid"), ("0dB", "#000000", "solid")]
    leg_ph = [("1.462kHz Target", "#a855f7", "cross"), ("Plant Gvf", "#94a3b8", "dashed"), ("Compensator Gc", "#fbbf24", "dotted"), ("Open-Loop T(s)", "#3b82f6", "solid"), ("Limit", "#ef4444", "dashed")]
    leg_step = [("Vout", "#10b981", "solid")]
    leg_pz = [("Poles", "#ef4444", "cross"), ("Zeros", "#10b981", "circle")]

    main = Div(cls="main-content")(
        Div(id="metrics-target")(Div("Loading system state...", cls="metric-card")),
        Div(cls="dashboard-grid")(
            make_html_chart("c-mag", "Loop Magnitude", "dB", "Hz", leg_mag),
            make_html_chart("c-phase", "Loop Phase", "Deg", "Hz", leg_ph),
            make_html_chart("c-step", "Step Response", "Vout", "Time (s)", leg_step),
            make_html_chart("c-pz", "Pole-Zero Map", "Im", "Re", leg_pz)
        ),
        Div(id="main-dashboard") # The script bucket target
    )
    return Body(sidebar, main, persistence_script)

@rt("/update")
def post(n:float, vout:float, pout:float, gain_trim:float, Td:float, GBW:float, r68:float, c71:float, c70:float, cout:float, resr:float, load_pct:float, fp_llc:float=3500, Q_llc:float=1.0, r65:float=51, c73:float=0, enable_lc:str=None, lf51:float=4.0, cout2:float=0.1, phase_view:str="math", **kwargs):
    lc_on = enable_lc == 'on' or enable_lc == 'true'
    return render_view(Lm, Lr, Cr, n, gain_trim, Td, GBW, r68, c71*1e-9, c70*1e-9, cout*1e-6, resr*1e-3, load_pct/100.0, phase_view, fp_llc, Q_llc, vout, pout, r65, c73*1e-9, lc_on, lf51*1e-6, cout2*1e-6)

def render_view(Lm, Lr, Cr, n, gain_trim, Td, GBW, r68, c71, c70, cout, resr, load, phase_view, fp_llc=3500, Q_llc=1.0, vout=20, pout=200, r65=51, c73=0, enable_lc=False, lf51=4e-6, cout2=0.1e-6):
    res = generate_analysis(Lm, Lr, Cr, n, Td, GBW, r68, c71, c70, cout, resr, load, fp_llc, Q_llc, phase_view, gain_trim, vout, pout, r65, c73, enable_lc, lf51, cout2)

    b_mag, b_ph, b_step, b_pz_json, fc, pm, gm, h_cl, r68, c71, c70 = res
    fc_s = f"{fc/1000:.2f} kHz" if fc else "N/A"
    pm_s = f"{pm:.1f}°" if pm else "N/A"
    gm_s = "∞ (Safe)" if gm == float('inf') else f"{gm:.1f} dB"
    
    # PZ Map special handling
    pz_fig = go.Figure(layout=get_base_layout(None, None, False))
    pz_fig.add_trace(go.Scatter(x=np.real(h_cl.poles), y=np.imag(h_cl.poles), mode='markers', name="Poles", uid="Poles", marker=dict(symbol='x', color='#ef4444')))
    pz_fig.add_trace(go.Scatter(x=np.real(h_cl.zeros), y=np.imag(h_cl.zeros), mode='markers', name="Zeros", uid="Zeros", marker=dict(symbol='circle-open', color='#10b981')))
    b_pz = pz_fig.to_json()

    fz_hz = 1.0 / (2 * np.pi * r68 * c71) if (r68 * c71) > 0 else 0
    fp_hz = 1.0 / (2 * np.pi * r68 * c70) if (r68 * c70) > 0 else 0
    commentary = ""
    commentary_style = "margin-bottom: 20px; padding: 10px; background: #1e293b; border-radius: 4px; font-weight: bold;"
    if fc is not None and fz_hz > 0:
        if fz_hz > fc / 3.0:
            commentary_style += " border-left: 4px solid #ef4444; color: #ef4444;"
            commentary = f"WARNING: Zero ({fz_hz:.1f} Hz) is too close to Crossover ({fc:.1f} Hz). Minimum 1/3 ratio recommended. Pole is at {fp_hz:.1f} Hz."
        else:
            commentary_style += " border-left: 4px solid #10b981; color: #10b981;"
            commentary = f"Zero ({fz_hz:.1f} Hz) placement is Good. Pole is at {fp_hz:.1f} Hz."

    return Div(id="main-dashboard")(
        Div(id="metrics-target", hx_swap_oob="true")(
            Div(cls="metrics-row")(
                Div(cls="metric-card")(Div("Crossover", cls="metric-label"), Div(fc_s, cls="metric-value")),
                Div(cls="metric-card")(Div("Phase Margin", cls="metric-label"), Div(pm_s, cls="metric-value")),
                Div(cls="metric-card")(Div("Gain Margin", cls="metric-label"), Div(gm_s, cls="metric-value")),
                Div(id="sync-top", cls="sync-spinner")("Syncing...")
            ),
            Div(commentary, style=commentary_style) if commentary else ""
        ),
        Script(f"""
            var cfg = {{responsive: false, displayModeBar: false}};
            var p = (n, j) => {{ var d = JSON.parse(j); Plotly.react(n, d.data, d.layout, cfg); }};
            p('c-mag', {json.dumps(b_mag)}); p('c-phase', {json.dumps(b_ph)}); 
            p('c-step', {json.dumps(b_step)}); p('c-pz', {json.dumps(b_pz)});
            
            var m_el = document.getElementById('c-mag');
            var p_el = document.getElementById('c-phase');
            if (m_el && p_el && !m_el._hasHoverSync) {{
                m_el._hasHoverSync = true; p_el._hasHoverSync = true;
                m_el.on('plotly_hover', d => Plotly.Fx.hover('c-phase', {{xval: d.points[0].x}}));
                m_el.on('plotly_unhover', () => Plotly.Fx.unhover('c-phase'));
                p_el.on('plotly_hover', d => Plotly.Fx.hover('c-mag', {{xval: d.points[0].x}}));
                p_el.on('plotly_unhover', () => Plotly.Fx.unhover('c-mag'));
            }}

            if(window.mermaid) mermaid.contentLoaded();
        """)
    )

serve(port=8000, reload=True)
