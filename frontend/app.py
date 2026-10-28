from __future__ import annotations
import numpy as np
import io
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError
# `streamlit run frontend/app.py` executes this file as a script, so use the
# sibling module import supported by Streamlit's script directory on sys.path.
from api import get_json, post_image
from visualization import (
    evaluation_metric_histogram,
    identical_image_quality_metrics,
    pixel_intensity_histogram,
)
from state import clear_upload, current_result, select_upload, store_result
from evaluation_summary import format_quality_mean, mean_group_metrics, mean_quality_metrics, metric_change

st.set_page_config(page_title="FPrint Studio", page_icon="◉", layout="wide", initial_sidebar_state="locked")
st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
NAV={"WORKSPACE":[("Overview","⌂"),("Enhance Image","＋"),("Image Comparison","⇄")],"EVALUATION":[("Model Evaluation","▤"),("Analytics","⌁")],"SYSTEM":[("Settings","⚙")]}
for key,value in (("page","Overview"),("api_url","http://127.0.0.1:5000"),("max_mb",12),("upload",None),("result",None),("session_runs",[])):
    if key not in st.session_state: st.session_state[key]=value

def fetch_health():
    try: return get_json(st.session_state.api_url,"/api/health",3),None
    except Exception as exc: return None,str(exc)

health,health_error=fetch_health(); model=(health or {}).get("model",{}); ready=bool((health or {}).get("ready"))
st.sidebar.markdown("<div class='brand'><div class='brand-mark'>◉</div><div><div class='brand-name'>FPrint Studio</div><div class='brand-sub'>Image Enhancement Lab</div></div></div>",unsafe_allow_html=True)
for group,items in NAV.items():
    st.sidebar.markdown(f"<div class='nav-label'>{group}</div>",unsafe_allow_html=True)
    for name,icon in items:
        if st.sidebar.button(f"{icon}   {name}",key=f"nav_{name}",width="stretch",type="primary" if st.session_state.page==name else "secondary"):
            st.session_state.page=name; st.rerun()

def header(eyebrow,title,subtitle):
    st.markdown(f"<div class='page-head'><div class='eyebrow'>{eyebrow}</div><div class='title'>{title}</div><div class='subtitle'>{subtitle}</div></div>",unsafe_allow_html=True)

def show_images():
    upload,result=st.session_state.upload,current_result(st.session_state)
    if not upload: st.info("Upload an image in Enhance Image to begin comparison."); return
    cols=st.columns(2)
    for col,label,data in ((cols[0],"Original",upload["data"]),(cols[1],"Enhanced",result["data"] if result else None)):
        with col:
            st.markdown(f"**{label}**")
            if data: st.image(data,width="stretch")
            else: st.info("Enhancement result is not available yet.")
    if result: st.caption(f"Input {upload['size'][0]} × {upload['size'][1]} px · Output {result['meta']['output_size']} px · Inference {result['meta']['seconds']*1000:.1f} ms")

def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        font={"family":"Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif","color":"#172033","size":12},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        margin={"l":45,"r":18,"t":28,"b":42},
        legend={"font":{"color":"#5B6475"},"bgcolor":"rgba(255,255,255,0.92)"},
    )
    fig.update_xaxes(color="#5B6475", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1")
    fig.update_yaxes(color="#5B6475", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1")
    return fig

page=st.session_state.page
if page=="Overview":
    header("BIOMETRIC IMAGE PROCESSING","Fingerprint Enhancement","Enhance ridge visibility, compare image quality, and evaluate your  .")
    try:
        analytics=get_json(st.session_state.api_url,"/api/analytics"); events=analytics["processing"]; runs=analytics["evaluations"]
    except Exception: events={"images_processed":0,"successful_runs":0,"mean_inference_seconds":None,"recent":[]}; runs=[]
    st.markdown("<p class='overview-intro'>Choose a workflow to begin. Enhancement processes an image you upload; evaluation measures the model on the configured held-out test split.</p>",unsafe_allow_html=True)
    enhance_col,evaluate_col=st.columns(2,gap="large")
    with enhance_col:
        with st.container(border=True,key="overview_enhancement_card"):
            st.markdown("""
            <div class="overview-card-content">
              <div class="overview-card-heading">
                <span class="overview-card-icon overview-card-icon--indigo" aria-hidden="true">
                  <svg viewBox="0 0 32 32" fill="none"><path d="M16 5.2c-6 0-10.8 4.6-10.8 10.3M16 9c-3.8 0-6.9 2.9-6.9 6.5 0 3.2-1.1 5.6-3 7.7M16 13c-1.7 0-3.1 1.3-3.1 3 0 4.4-1.2 7.8-3.1 10.2M21.5 7.9c2.9 1.8 4.9 4.8 4.9 8.2 0 4-1.1 7.5-3 10.1M16 5.2c2 0 3.9.5 5.5 1.5M20.5 15.3c0 4.1-.7 7.8-2 10.8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
                </span>
                <span class="overview-card-label">IMAGE PROCESSING</span>
              </div>
              <h2 class="overview-card-title">Enhance Fingerprint</h2>
              <p class="overview-card-description">Upload a fingerprint image, improve ridge visibility, and compare the enhanced output with the original.</p>
            </div>
            """,unsafe_allow_html=True)
            if st.button("Start Enhancement",key="overview_start_enhancement",type="primary",width="stretch"):
                st.session_state.page="Enhance Image"; st.rerun()
    with evaluate_col:
        with st.container(border=True,key="overview_evaluation_card"):
            st.markdown("""
            <div class="overview-card-content">
              <div class="overview-card-heading">
                <span class="overview-card-icon overview-card-icon--teal" aria-hidden="true">
                  <svg viewBox="0 0 32 32" fill="none"><path d="M6 25.5V17M13 25.5V11M20 25.5V6.5M27 25.5v-12" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><path d="M4.5 26.5h23" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
                </span>
                <span class="overview-card-label overview-card-label--teal">MODEL PERFORMANCE</span>
              </div>
              <h2 class="overview-card-title">Evaluate Model</h2>
              <p class="overview-card-description">Run your model against the paired held-out test dataset and analyze PSNR, SSIM, and MSE.</p>
            </div>
            """,unsafe_allow_html=True)
            if st.button("Run Evaluation",key="overview_run_evaluation",type="primary",width="stretch"):
                st.session_state.page="Model Evaluation"; st.rerun()

    st.markdown("<div class='overview-section-heading'><span class='overview-section-kicker'>THE WORKFLOW</span><h2>How it works</h2></div>",unsafe_allow_html=True)
    step_cols=st.columns(3,gap="medium")
    steps=(
        ("01","Upload a fingerprint image","Choose a supported fingerprint image to preview."),
        ("02","Run trained enhancement","The configured PyTorch model processes the image."),
        ("03","Compare and evaluate","Inspect before and after, then review test metrics."),
    )
    for col,(number,title,description) in zip(step_cols,steps):
        with col:
            st.markdown(f"<div class='overview-step'><span class='overview-step-number'>{number}</span><div><div class='overview-step-title'>{title}</div><div class='overview-step-description'>{description}</div></div></div>",unsafe_allow_html=True)

elif page=="Enhance Image":
    header("WORKSPACE","Enhance Image","Enhance fingerprint images using the trained PyTorch model with preprocessing configured to match the model's training pipeline.")
    uploaded=st.file_uploader("Choose a fingerprint image",type=["png","jpg","jpeg","tif","tiff","bmp"],help=f"Maximum {st.session_state.max_mb} MB")
    if uploaded:
        data=uploaded.getvalue()
        if len(data)>st.session_state.max_mb*1024*1024:
            clear_upload(st.session_state); st.error(f"File exceeds the {st.session_state.max_mb} MB upload limit.")
        else:
            try:
                im=Image.open(io.BytesIO(data)); im.seek(0); im.load(); im=im.convert("L")
                select_upload(st.session_state,data,uploaded.name,im.size)
            except (OSError,UnidentifiedImageError):
                clear_upload(st.session_state); st.error("This file could not be decoded as an image.")
    up=st.session_state.upload
    if up:
        st.markdown(f"**{up['filename']}** · {up['size'][0]} × {up['size'][1]} px"); st.image(up["data"],caption="Input image",width=650)
        result=current_result(st.session_state)
        if st.button("Enhance fingerprint",type="primary",disabled=not ready or result is not None):
            try:
                with st.spinner("Running model inference…"): out,meta=post_image(st.session_state.api_url,up["data"],up["filename"])
                store_result(st.session_state,out,meta); st.session_state.session_runs.append({"filename":up["filename"],"seconds":meta["seconds"]}); st.success(f"Enhancement complete.")
            except Exception as exc: st.error(f"Enhancement failed: {exc}")
        if result is not None: st.caption("This image already has an enhancement result. Open Image Comparison to reuse it.")
        if not ready: st.warning("The trained model is unavailable. Configure a compatible checkpoint and restart Flask.")
    result=current_result(st.session_state)
    if result:
        st.divider(); show_images(); st.download_button("Download enhanced PNG",result["data"],file_name="enhanced_fingerprint.png",mime="image/png")
        st.caption("Output is 8-bit grayscale at the uploaded image dimensions. Pixel statistics describe intensity, not fingerprint quality.")

elif page=="Image Comparison":
    header("WORKSPACE","Image Comparison","Inspect model output beside its input with aspect ratio preserved."); show_images()
    up,res=st.session_state.upload,st.session_state.result
    if up and res:
        with st.container(border=True):
            st.markdown("**Pixel intensity distribution** · Brightness is not a measure of enhancement quality.")
            original=np.asarray(Image.open(io.BytesIO(up["data"])).convert("L")).ravel(); enhanced=np.asarray(Image.open(io.BytesIO(res["data"])).convert("L")).ravel()
            fig=style_chart(pixel_intensity_histogram(original,enhanced)); fig.update_layout(height=330); st.plotly_chart(fig,width="stretch")

elif page=="Model Evaluation":
    header("EVALUATION","Model Evaluation","Evaluate the existing held-out test split using deterministic synthetic degradation.")
    st.info(
        """
        **Understanding the evaluation metrics**

        - **PSNR (Peak Signal-to-Noise Ratio):** Measures how closely the enhanced image
        matches the clean reference. Higher values generally indicate lower pixel-level error.

        - **SSIM (Structural Similarity Index):** Measures similarity in image structure,
        contrast, and luminance. Values closer to 1 indicate greater structural similarity.

        - **MSE (Mean Squared Error):** Measures the average squared pixel difference
        between the enhanced image and its clean reference. Lower values indicate less error.

        - **Inference Time:** Measures how long the model takes to enhance one image.
        Lower values indicate faster processing.

        PSNR, SSIM, and MSE are calculated against paired clean reference images.
        """
    )

    if st.button("Start held-out evaluation",type="primary",disabled=not ready):
        try:
            response=requests.post(st.session_state.api_url.rstrip("/")+"/api/evaluate",timeout=10); response.raise_for_status(); st.session_state.eval_job=response.json()["job_id"]
        except Exception as exc: st.error(f"Could not start evaluation: {exc}")
    job_id=st.session_state.get("eval_job")
    if job_id:
        try:
            job=get_json(st.session_state.api_url,"/api/evaluate/"+job_id,5); st.progress(job.get("progress",0)/100,text=job.get("status","running"))
            if job.get("status")=="completed": st.success(f"Completed: {job['result']['sample_count']} samples; {job['result']['failed_count']} failed."); st.json(job["result"]["summary"])
            if job.get("status")=="failed": st.error(job.get("error"))
            if job.get("status")=="running": st.caption("Refresh the page to check progress.")
        except Exception as exc: st.error(str(exc))
    try: runs=get_json(st.session_state.api_url,"/api/analytics")["evaluations"]
    except Exception as exc: runs=[]; st.warning(f"Evaluation service unavailable: {exc}")

    selected_run=None
    if not runs:
        st.caption("No persisted evaluation runs yet.")
    else:
        run_by_id={str(run["run_id"]):run for run in runs}
        run_ids=list(run_by_id)
        run_labels={run_id:f"{run_id}  ?  {str(run_by_id[run_id].get('created_at','date unavailable'))[:10]}" for run_id in run_ids}
        selected_id=st.selectbox(
            "Evaluation run",
            options=run_ids,
            index=len(run_ids)-1,
            format_func=lambda run_id:run_labels[run_id],
            key="evaluation_run_selection",
        )
        selected_run=run_by_id[selected_id]
        st.subheader(f"Run {selected_id}")
        rows=selected_run.get("rows",[])
        if isinstance(rows,list) and rows:
            frame=pd.json_normalize(rows,sep="_")
            st.dataframe(frame,width="stretch",hide_index=True)
            st.download_button("Export per-image CSV",frame.to_csv(index=False),file_name=f"{selected_id}.csv",mime="text/csv",key=f"csv_{selected_id}")
        else:
            st.info("This evaluation run contains no per-image results.")

    summary_rows=selected_run.get("rows",[]) if selected_run else []
    summary=mean_quality_metrics(summary_rows)
    summary_cards=(
        ("PSNR (dB)",summary["psnr_db"],"Higher generally indicates lower pixel-level error.","psnr_db"),
        ("SSIM",summary["ssim"],"Higher indicates greater structural similarity to the clean reference.","ssim"),
        ("MSE",summary["mse"],"Lower indicates smaller pixel-level differences.","mse"),
    )
    cards_html="".join(
        f"<article class=\"evaluation-summary-card\"><div class=\"evaluation-summary-label\">{label}</div><div class=\"evaluation-summary-value\">{format_quality_mean(value,metric)}</div><div class=\"evaluation-summary-description\">{description}</div></article>"
        for label,value,description,metric in summary_cards
    )
    st.markdown(
        f"""<section class="evaluation-summary-box" aria-labelledby="evaluation-summary-title">
          <div class="evaluation-summary-heading">
            <span class="evaluation-summary-kicker">SELECTED RUN</span>
            <h2 id="evaluation-summary-title">Overall Evaluation Summary</h2>
            <p>Arithmetic mean of available per-image reference-based metrics.</p>
          </div>
          <div class="evaluation-summary-grid">{cards_html}</div>
        </section>""",
        unsafe_allow_html=True,
    )

elif page=="Analytics":
    header("EVALUATION","Analytics","Research views derived only from persisted evaluation results.")
    try: runs=get_json(st.session_state.api_url,"/api/analytics")["evaluations"]
    except Exception as exc: runs=[]; st.error(str(exc))
    if not runs: st.info("No evaluation data yet. Run a held-out evaluation to populate these charts.")
    else:
        run_by_id={str(run["run_id"]):run for run in runs}
        run_ids=list(run_by_id)
        run_labels={run_id:f"{run_id}  ?  {run_by_id[run_id].get('created_at','date unavailable')[:10]}" for run_id in run_ids}
        selected_ids=st.multiselect(
            "Evaluation runs",
            options=run_ids,
            default=[run_ids[-1]],
            format_func=lambda run_id:run_labels[run_id],
            help="Select one run to inspect it, or select multiple runs to compare their distributions.",
        )
        if not selected_ids:
            st.info("Select at least one evaluation run to display analytics.")
        else:
            selected_runs=[run_by_id[run_id] for run_id in selected_ids]
            selected_rows=pd.concat(
                [pd.DataFrame(run.get("rows",[])).assign(run_id=str(run["run_id"])) for run in selected_runs],
                ignore_index=True,
            ) if any(run.get("rows") for run in selected_runs) else pd.DataFrame()

            if len(selected_runs)>1 and identical_image_quality_metrics(selected_runs):
                latency_signatures=[
                    [(row.get("image"),row.get("inference_ms")) for row in run.get("rows",[])]
                    for run in selected_runs
                ]
                latency_note=" Inference latency differs and is compared separately below." if any(
                    values!=latency_signatures[0] for values in latency_signatures[1:]
                ) else " Inference latency is operational data and is compared separately below."
                st.info("The selected runs have identical per-image PSNR, SSIM, and MSE values in the same image order."+latency_note)

            if selected_rows.empty:
                st.info("The selected runs do not contain per-image rows to chart.")
            else:
                st.markdown("### Image-quality distributions")
                quality_cols=st.columns(3,gap="medium")
                for col,(metric,label) in zip(quality_cols,(("psnr_db","PSNR (dB)"),("ssim","SSIM"),("mse","MSE"))):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{label}**")
                            fig=style_chart(evaluation_metric_histogram(selected_runs,metric))
                            fig.update_layout(height=340,xaxis_title=label,yaxis_title="Share of images")
                            st.plotly_chart(fig,width="stretch")


                if "psnr_db" in selected_rows:
                    with st.container(border=True):
                        st.markdown("**Per-image PSNR comparison**")
                        plot_rows=selected_rows.dropna(subset=["psnr_db"])
                        if plot_rows.empty: st.caption("PSNR values are unavailable for the selected rows.")
                        else:
                            fig=style_chart(px.scatter(plot_rows,x="image",y="psnr_db",color="run_id",hover_data=[col for col in ("ssim","mse","inference_ms") if col in plot_rows]))
                            fig.update_layout(height=390,xaxis_title="Image",yaxis_title="PSNR (dB)",legend_title_text="Evaluation run")
                            st.plotly_chart(fig,width="stretch")
                st.download_button("Export selected evaluation rows",selected_rows.to_csv(index=False),file_name="evaluation_analytics.csv",mime="text/csv")

elif page=="Settings":
    header("SYSTEM","Settings","Configure the dashboard connection and display preferences.")
    with st.form("settings"):
        url=st.text_input("Backend base URL",st.session_state.api_url); limit=st.number_input("Maximum upload size (MB)",1,100,st.session_state.max_mb)
        preview=st.selectbox("Preview fit",["Container width","Original pixel dimensions"]); fmt=st.selectbox("Download format",["PNG"])
        if st.form_submit_button("Save settings"):
            st.session_state.api_url=url.rstrip("/"); st.session_state.max_mb=int(limit); st.success("Settings saved for this session.")
    st.caption(f"Preview: {preview} · Download: {fmt}")
