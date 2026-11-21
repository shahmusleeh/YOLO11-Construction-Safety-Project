import gradio as gr
import cv2
from ultralytics import YOLO
from pathlib import Path

# Load your trained model weights
model_path = 'best.pt'
detector = YOLO(model_path)

def detect_image(image, conf):
    if image is None:
        return None, 'Please upload an image.'
    result = detector.predict(source=image, conf=float(conf), imgsz=640, verbose=False)[0]
    rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    counts = {}
    for cls in result.boxes.cls.cpu().numpy().astype(int):
        name = detector.names[int(cls)]
        counts[name] = counts.get(name, 0) + 1
   details = "\n".join(
    f"{name}: {n}"
    for name, n in sorted(counts.items())
) or "No objects detected."
violation_found = any(
    name in SAFETY_VIOLATIONS
    for name in counts
)
warning = (
    "\n\n⚠️ PPE VIOLATION DETECTED!"
    if violation_found
    else ""
)

return rgb, details + warning
    "\n\n⚠️ PPE VIOLATION DETECTED!"
    if any(name in SAFETY_VIOLATIONS for name in counts)
    else ""
)

' + '
'.join(warnings) if warnings else '')

with gr.Blocks(title='Smart Construction Safety Monitoring') as demo:
    gr.Markdown("# Smart Construction Safety Monitoring\nUpload a construction-site photo. The trained YOLO model detects annotated objects and reports class counts. **PPE warnings are based on predicted NO-* labels and may be inaccurate.**")
    with gr.Row():
        input_image = gr.Image(type='numpy', label='Upload construction site image')
        output_image = gr.Image(label='YOLO detections')
    conf_slider = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label='Confidence threshold')
    summary = gr.Textbox(label='Detected objects and warning labels', lines=8)
    gr.Button('Analyze image', variant='primary').click(
        detect_image, inputs=[input_image, conf_slider], outputs=[output_image, summary]
    )

if __name__ == '__main__':
    demo.launch()
