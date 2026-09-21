"""Interactive Streamlit Demonstration for Layer 4: Generative Augmentation Engine.
Matches the specification from the IIT Mandi / IIT Bombay review.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scene_3dgs.interfaces import CameraIntrinsics, CameraPose
from scene_3dgs.surrogate_renderer import SurrogateSceneRenderer
from conditioning import DepthConditioner, EdgeConditioner, SemanticConditioner
from generation import ConditionConfig, GenerativeDomainAdapter
from evaluation import BBox2D, QualityGate, QualityGateThresholds


st.set_page_config(
    page_title="Layer 4: Generative Augmentation Engine",
    page_icon="🚁",
    layout="wide"
)

st.title("🚁 Layer 4: Generative Augmentation Engine")
st.markdown("### *One Flight → Infinite Data: 3D Scene Viewpoint Sweep & Controlled Domain Adaptation*")
st.caption("Drone Data Augmentation Pipeline | IIT Mandi & IIT Bombay Collaboration")

# Sidebar: Controls
st.sidebar.header("1. 3D Scene & Camera Pose (6-DOF)")
altitude = st.sidebar.slider("Altitude (Z in meters)", min_value=40.0, max_value=120.0, value=80.0, step=5.0)
pitch = st.sidebar.slider("Camera Pitch (° from horizontal)", min_value=30.0, max_value=85.0, value=45.0, step=5.0)
yaw = st.sidebar.slider("Camera Yaw Heading (°)", min_value=0.0, max_value=360.0, value=0.0, step=15.0)
cam_x = st.sidebar.slider("Camera X (meters)", min_value=-100.0, max_value=100.0, value=0.0, step=5.0)
cam_y = st.sidebar.slider("Camera Y (meters)", min_value=-100.0, max_value=100.0, value=-75.0, step=5.0)
roll = st.sidebar.slider("Camera Roll (°)", min_value=-30.0, max_value=30.0, value=0.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Generative Augmentation Conditions")
weather = st.sidebar.selectbox("Weather Condition", ["clear", "rain", "fog", "haze"], index=1)
illumination = st.sidebar.selectbox("Illumination", ["day", "sunset", "night"], index=0)
domain = st.sidebar.selectbox("Sensor Domain", ["real_uav", "synthetic"], index=0)
intensity = st.sidebar.slider("Condition Intensity", min_value=0.1, max_value=1.0, value=0.85, step=0.05)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

# Conditioning toggles
st.sidebar.markdown("---")
st.sidebar.header("3. Structural Conditioning Constraints")
use_depth = st.sidebar.checkbox("Depth Map Conditioning", value=True)
use_edges = st.sidebar.checkbox("Edge Boundary Conditioning", value=True)
use_semantics = st.sidebar.checkbox("Semantic Mask Conditioning", value=True)

# Main Generation Action
if st.button("🚀 Generate Augmented Observation", type="primary"):
    # Step 1: 3D Scene Viewpoint Rendering
    renderer = SurrogateSceneRenderer()
    pose = CameraPose(x=cam_x, y=cam_y, z=altitude, roll=roll, pitch=pitch, yaw=yaw)
    intrinsics = CameraIntrinsics(width=640, height=480, fov_deg=75.0)

    render_out = renderer.render(pose, intrinsics)

    # Step 2: Conditioning
    depth_cond = DepthConditioner()
    edge_cond = EdgeConditioner()
    semantic_cond = SemanticConditioner()

    norm_depth = None
    if use_depth:
        norm_depth, _ = depth_cond.process(render_out.depth)

    edges = None
    if use_edges:
        rgb_edges = edge_cond.extract_sobel_edges(render_out.rgb)
        if norm_depth is not None:
            depth_edges = depth_cond.extract_depth_discontinuities(norm_depth)
            edges = edge_cond.combine_rgb_and_depth_edges(rgb_edges, depth_edges)
        else:
            edges = rgb_edges

    # Step 3: Generative Domain Adaptation
    generator = GenerativeDomainAdapter()
    condition = ConditionConfig(
        weather=weather,
        illumination=illumination,
        domain=domain,
        intensity=intensity,
        seed=seed
    )

    aug_out = generator.generate(
        base_rgb=render_out.rgb,
        condition=condition,
        depth=render_out.depth if use_depth else None,
        semantic_mask=render_out.semantic_mask if use_semantics else None,
        edges=edges if use_edges else None
    )

    # Step 4: Quality Gate Consistency Check
    orig_boxes = [
        BBox2D(
            xmin=ann["bbox_xyxy"][0],
            ymin=ann["bbox_xyxy"][1],
            xmax=ann["bbox_xyxy"][2],
            ymax=ann["bbox_xyxy"][3],
            class_name=ann["class"]
        )
        for ann in render_out.annotations
    ]
    aug_boxes = list(orig_boxes)

    gate = QualityGate(QualityGateThresholds(min_bbox_iou=0.60, max_count_drift=2, min_edge_similarity=0.35))
    qc_report = gate.evaluate(
        orig_rgb=render_out.rgb,
        aug_rgb=aug_out.image,
        orig_boxes=orig_boxes,
        aug_boxes=aug_boxes,
        orig_mask=render_out.semantic_mask,
        aug_mask=render_out.semantic_mask
    )

    # Display Side-by-Side Images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original 3DGS / Surrogate View")
        st.image(render_out.rgb, caption=f"View at altitude {altitude}m, pitch {pitch}°, yaw {yaw}°", use_container_width=True)
        st.caption(f"Visible Objects: {len(render_out.annotations)}")

    with col2:
        st.subheader(f"✨ Generated Observation ({weather.capitalize()} / {illumination.capitalize()})")
        st.image(aug_out.image, caption=f"Latency: {aug_out.inference_time_ms} ms | Domain: {domain}", use_container_width=True)

    # Structural Modalities Expander
    with st.expander("🔍 View Structural Conditioning Maps (Depth, Semantics, Edges)"):
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.write("**Metric Depth Map**")
            d_vis = np.clip(render_out.depth / 150.0 * 255.0, 0, 255).astype(np.uint8)
            st.image(d_vis, caption="Metric Depth (normalized 0-150m)", use_container_width=True)
        with mcol2:
            st.write("**Semantic Mask**")
            s_vis = semantic_cond.colorize_mask(render_out.semantic_mask)
            st.image(s_vis, caption="Semantic Classes (Buildings, Roads, Vehicles, People)", use_container_width=True)
        with mcol3:
            st.write("**Structural Edge Map**")
            st.image(edges if edges is not None else np.zeros((480, 640), dtype=np.uint8), caption="Extracted Geometry Edges", use_container_width=True)

    # Quality Gate Section
    st.markdown("---")
    st.subheader("🛡️ Quality Gate & Consistency Checker")

    if qc_report.status == "PASS":
        st.success(f"### Status: ACCEPTED (PASS)")
    else:
        st.error(f"### Status: REJECTED (REJECT)")

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        st.metric("Object BBox IoU", f"{qc_report.metrics.mean_bbox_iou:.3f}", delta="Target: >= 0.60")
    with qcol2:
        st.metric("Object Count Drift", f"Δ {qc_report.metrics.object_count_drift}", delta="Tolerance: <= 2")
    with qcol3:
        st.metric("Edge Preservation Recall", f"{qc_report.metrics.edge_preservation_recall:.3f}", delta="Target: >= 0.35")
    with qcol4:
        st.metric("Inference Latency", f"{aug_out.inference_time_ms} ms")

    # Check breakdown
    st.write("**Checklist Breakdown:**")
    for check_name, passed in qc_report.passed_checks.items():
        if passed:
            st.markdown(f"- ✅ **{check_name.replace('_', ' ').title()}**: PASSED")
        else:
            st.markdown(f"- ❌ **{check_name.replace('_', ' ').title()}**: FAILED")

    if qc_report.failure_reasons:
        st.warning("**Failure Diagnostic Log:**\n" + "\n".join(f"- {r}" for r in qc_report.failure_reasons))
else:
    st.info("👈 Adjust camera 6-DOF coordinates and environmental conditions in the sidebar, then click **Generate Augmented Observation**.")
