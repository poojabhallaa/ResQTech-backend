# import streamlit as st
# import cv2
# import os
# import tempfile
# from ultralytics import YOLO
# import google.generativeai as genai
# from fpdf import FPDF
# from collections import Counter

# # Gemini setup
# genai.configure(api_key="AIzaSyDS9uZKp5l6xOdgqLVUxcxjp8QvDwNY69M")
# model = genai.GenerativeModel("gemini-2.0-flash")

# # YOLO detection
# def detect_objects_yolo(file_path, is_video=False):
#     yolo_model = YOLO("yolov8n.pt")  # Replace with custom model if needed
#     detections = []

#     if is_video:
#         cap = cv2.VideoCapture(file_path)
#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret or frame_count > 30:
#                 break
#             temp_frame_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_count}.jpg")
#             cv2.imwrite(temp_frame_path, frame)
#             result = yolo_model(temp_frame_path)
#             summary = summarize_detection(result[0].names, result[0].boxes.cls.cpu().numpy())
#             detections.append((temp_frame_path, summary))
#             frame_count += 5
#         cap.release()
#     else:
#         result = yolo_model(file_path)
#         summary = summarize_detection(result[0].names, result[0].boxes.cls.cpu().numpy())
#         detections.append((file_path, summary))

#     return detections

# # Summarize YOLO results
# def summarize_detection(names, class_ids):
#     count = Counter(class_ids)
#     summary = []
#     for class_id, num in count.items():
#         summary.append(f"{int(num)} {names[int(class_id)]}(s)")
#     return ", ".join(summary)

# # 🔥 Advanced Gemini prompt for detailed disaster report
# def generate_report_with_gemini(detection_summary):
#     prompt = f"""
# You are a disaster response analyst. Based on the following visual detection summary:
# '{detection_summary}', create a professional, structured report suitable for submission to government agencies, news media, or NGOs.

# Your report must include:
# 1. **Type of Disaster**
# 2. **Probable Location** (if identifiable)
# 3. **Scale of Damage**
# 4. **Infrastructure Damage**
# 5. **Estimated Casualties/Injuries**
# 6. **Environmental Impact**
# 7. **Immediate Needs**
# 8. **Involved Agencies (if visible)**
# 9. **Time/Date (if inferred visually)**
# 10. **Contributing Factors**
# 11. **Summary of Situation**

# Write in formal, factual language. Do not ask for more input. Avoid speculative or vague language unless necessary. This will be used as an official report.
# """
#     response = model.generate_content(prompt)
    
#     # Handling the model's response properly
#     if response and hasattr(response, 'text'):
#         return response.text.strip()
#     else:
#         return "AI could not generate a report. Please check the inputs."

# # 📄 PDF creation
# def create_pdf_report(reports):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     for idx, (image_path, summary, ai_report) in enumerate(reports):
#         pdf.add_page()
#         pdf.set_font("Arial", size=12)
#         if image_path.lower().endswith((".jpg", ".png", ".jpeg")):
#             pdf.image(image_path, w=150)
#         pdf.ln(10)
#         pdf.multi_cell(0, 10, f"Detection Summary: {summary}")
#         pdf.ln(5)
#         pdf.multi_cell(0, 10, f"AI Report:\n{ai_report}")
#     output_path = os.path.join(tempfile.gettempdir(), "Disaster_Report.pdf")
#     pdf.output(output_path)
#     return output_path

# # 🚀 Streamlit App
# st.title("🛸 Disaster Area Detection & AI Summary")
# st.markdown("Upload images or videos of disaster zones. Get AI-generated reports detecting trapped humans, damage, and more.")

# # Allow uploading both images and videos
# uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
# uploaded_videos = st.file_uploader("Upload Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

# generate = st.button("🔍 Generate Disaster Report")

# if generate:
#     all_reports = []
#     if not uploaded_images and not uploaded_videos:
#         st.warning("Please upload at least one image or video.")
#     else:
#         with st.spinner("Processing files..."):
#             # Process images
#             if uploaded_images:
#                 for img_file in uploaded_images:
#                     temp_path = os.path.join(tempfile.gettempdir(), img_file.name)
#                     with open(temp_path, "wb") as f:
#                         f.write(img_file.read())
#                     detections = detect_objects_yolo(temp_path)
#                     for file_path, detection_summary in detections:
#                         ai_summary = generate_report_with_gemini(detection_summary)
#                         all_reports.append((file_path, detection_summary, ai_summary))

#             # Process videos
#             if uploaded_videos:
#                 for vid_file in uploaded_videos:
#                     temp_path = os.path.join(tempfile.gettempdir(), vid_file.name)
#                     with open(temp_path, "wb") as f:
#                         f.write(vid_file.read())
#                     detections = detect_objects_yolo(temp_path, is_video=True)
#                     for file_path, detection_summary in detections:
#                         ai_summary = generate_report_with_gemini(detection_summary)
#                         all_reports.append((file_path, detection_summary, ai_summary))

#         st.success("Report generation complete!")

#         # Display the AI-generated report on the screen
#         for file_path, detection_summary, ai_summary in all_reports:
#             st.subheader(f"Report for {os.path.basename(file_path)}")
#             st.write(f"**Detection Summary:** {detection_summary}")
#             st.write(f"**AI Report:**\n{ai_summary}")

#         # Generate and provide PDF download link
#         pdf_path = create_pdf_report(all_reports)
#         with open(pdf_path, "rb") as pdf_file:
#             st.download_button("📄 Download PDF Report", data=pdf_file, file_name="Disaster_AI_Report.pdf")



import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import google.generativeai as genai
from fpdf import FPDF
from collections import Counter
import unicodedata

# Gemini setup
genai.configure(api_key="AIzaSyDS9uZKp5l6xOdgqLVUxcxjp8QvDwNY69M")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-2.0-flash")

# YOLO Detection
def detect_objects_yolo(file_path, is_video=False):
    yolo_model = YOLO("yolov8n.pt")
    detections = []

    if is_video:
        cap = cv2.VideoCapture(file_path)
        frame_count = 0
        max_frames = 2
        processed = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // max_frames if total_frames >= max_frames else 1

        while processed < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            temp_frame_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            result = yolo_model(temp_frame_path)
            summary = summarize_detection(result[0].names, result[0].boxes.cls.cpu().numpy())
            detections.append((temp_frame_path, summary))

            frame_count += step
            processed += 1

        cap.release()
    else:
        result = yolo_model(file_path)
        summary = summarize_detection(result[0].names, result[0].boxes.cls.cpu().numpy())
        detections.append((file_path, summary))

    return detections

# Detection Summary
def summarize_detection(names, class_ids):
    count = Counter(class_ids)
    summary = []
    for class_id, num in count.items():
        summary.append(f"{int(num)} {names[int(class_id)]}(s)")
    return ", ".join(summary)

# Gemini Report Generation (Simplified Prompt)
def generate_report_with_gemini(detection_summary):
    prompt = f"You are a disaster analyst. Analyze the disaster from the following detection summary: '{detection_summary}' and generate a detailed situation report."
    response = model.generate_content(prompt)
    return response.text.strip() if response and hasattr(response, 'text') else "AI could not generate a report."

# RL Agent Simulation
def generate_rl_actions(detection_summary):
    # Simulated logic for reinforcement learning output
    actions = [
        "Deploy emergency response team to assess safety.",
        "Send rescue units with boats and medical aid.",
        "Distribute food and clean water supplies.",
        "Establish temporary shelters for displaced people.",
        "Use drones to continue aerial surveillance.",
        "Coordinate with local NGOs for relief distribution.",
        "Request media to broadcast safety and aid info."
    ]
    return actions[:4]  # Return top 4 suggested actions for brevity

# Clean Unicode for PDF
def clean_text_for_pdf(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# PDF Report Generator
def create_pdf_report(reports):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for idx, (image_path, summary, ai_report, rl_actions) in enumerate(reports):
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        if image_path.lower().endswith((".jpg", ".png", ".jpeg")):
            pdf.image(image_path, w=150)
        pdf.ln(10)

        pdf.multi_cell(0, 10, f"Detection Summary: {clean_text_for_pdf(summary)}")
        pdf.ln(5)

        pdf.multi_cell(0, 10, "AI Report:")
        pdf.multi_cell(0, 10, clean_text_for_pdf(ai_report))
        pdf.ln(5)

        pdf.multi_cell(0, 10, "RL Agent Suggested Actions:")
        for action in rl_actions:
            pdf.multi_cell(0, 10, f"- {clean_text_for_pdf(action)}")
    
    output_path = os.path.join(tempfile.gettempdir(), "Disaster_Report.pdf")
    pdf.output(output_path)
    return output_path

# Streamlit UI
st.title("🛸 Disaster Detection + AI Report + RL Action Advisor")
st.markdown("Upload images or videos of disaster zones. Get AI-generated reports and simulated reinforcement learning suggestions.")

uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_videos = st.file_uploader("Upload Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if st.button("🔍 Generate Full Report"):
    all_reports = []
    if not uploaded_images and not uploaded_videos:
        st.warning("Please upload at least one image or video.")
    else:
        with st.spinner("Processing..."):
            # Images
            if uploaded_images:
                for img_file in uploaded_images:
                    temp_path = os.path.join(tempfile.gettempdir(), img_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(img_file.read())
                    detections = detect_objects_yolo(temp_path)
                    for file_path, detection_summary in detections:
                        ai_summary = generate_report_with_gemini(detection_summary)
                        rl_suggestions = generate_rl_actions(detection_summary)
                        all_reports.append((file_path, detection_summary, ai_summary, rl_suggestions))

            # Videos
            if uploaded_videos:
                for vid_file in uploaded_videos:
                    temp_path = os.path.join(tempfile.gettempdir(), vid_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(vid_file.read())
                    detections = detect_objects_yolo(temp_path, is_video=True)
                    for file_path, detection_summary in detections:
                        ai_summary = generate_report_with_gemini(detection_summary)
                        rl_suggestions = generate_rl_actions(detection_summary)
                        all_reports.append((file_path, detection_summary, ai_summary, rl_suggestions))

        st.success("✔ Report generation complete!")

        for file_path, detection_summary, ai_summary, rl_suggestions in all_reports:
            st.subheader(f"📷 {os.path.basename(file_path)}")
            st.write(f"**Detection Summary:** {detection_summary}")
            st.write("**AI Report:**")
            st.write(ai_summary)
            st.write("**RL Agent Suggestions:**")
            for item in rl_suggestions:
                st.markdown(f"- {item}")

        pdf_path = create_pdf_report(all_reports)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("📄 Download PDF Report", data=pdf_file, file_name="Disaster_AI_Report.pdf")



# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import tempfile
# import os
# from PIL import Image

# # Load YOLOv8 Model (Pre-trained)
# @st.cache_resource
# def load_model():
#     model = YOLO('yolov8n.pt')  # You can use different model variants like 'yolov8s.pt', 'yolov8m.pt'
#     return model

# model = load_model()

# # Function to process video frames, display them, and detect objects
# def process_video(video_file):
#     # Create a placeholder for displaying the video
#     video_placeholder = st.empty()
    
#     # Create a status text placeholder
#     status_text = st.empty()
    
#     # Create a temporary file for the video
#     temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
    
#     # Save the uploaded video to a temporary file
#     with open(temp_video_path, "wb") as f:
#         f.write(video_file.getbuffer())
    
#     # Open the saved video file with OpenCV
#     cap = cv2.VideoCapture(temp_video_path)
    
#     if not cap.isOpened():
#         st.error(f"Error: Could not open video {video_file.name}.")
#         return
    
#     # Initialize variables
#     frame_count = 0
#     status_text.text("Processing video... Please wait.")
    
#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Create a progress bar
#     progress_bar = st.progress(0)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Update progress bar
#         frame_count += 1
#         progress = int(frame_count / total_frames * 100)
#         progress_bar.progress(progress)
        
#         # Perform object detection on the frame
#         results = model(frame)
#         detections = results[0].boxes  # Access detection results from the first output
        
#         detected_labels = []
        
#         for detection in detections:
#             # Access the detection box and additional information (xywh, confidence, class_id)
#             xywh = detection.xywh[0]
#             confidence = detection.conf[0]
#             class_id = int(detection.cls[0])  # Convert class ID to integer
#             label = model.names[class_id]
            
#             if confidence > 0.4:  # Confidence threshold
#                 detected_labels.append(label)
                
#                 # Extract bounding box coordinates
#                 x_center, y_center, w, h = xywh
                
#                 # Draw bounding box for detected objects
#                 color = (0, 255, 0)  # Green color for detected objects
#                 start_point = (int(x_center - w/2), int(y_center - h/2))
#                 end_point = (int(x_center + w/2), int(y_center + h/2))
                
#                 # Draw the rectangle (bounding box)
#                 cv2.rectangle(frame, start_point, end_point, color, 2)
                
#                 # Add label and confidence next to the bounding box
#                 text = f"{label} {confidence:.2f}"
#                 cv2.putText(frame, text, 
#                             (start_point[0], start_point[1] - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # Convert the frame from BGR to RGB (for displaying in Streamlit)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Display the frame with bounding boxes in Streamlit
#         video_placeholder.image(frame_rgb, caption=f"Frame {frame_count}/{total_frames}", use_container_width=True)
        
#         # To simulate real-time processing but not too fast
#         # Adjust the sleep time based on video fps for a more natural display
#         import time
#         time.sleep(1/fps)  # Sleep to simulate real-time playback
    
#     # Release video capture and delete temporary file
#     cap.release()
#     os.remove(temp_video_path)
    
#     status_text.success("Video analysis complete.")
    
# # Streamlit User Interface
# st.title("Smart Disaster Detection System")
# st.markdown("### Upload your disaster-related CCTV video footage to detect objects")

# # Create tabs for better organization
# tab1, tab2 = st.tabs(["Upload Video", "Results"])

# with tab1:
#     # User input fields
#     disaster_video = st.file_uploader("Upload disaster-related video footage (e.g., fire, flood, damage, people)", type=["mp4", "avi", "mov"])
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         start_button = st.button("Start Detection", type="primary")
    
#     with col2:
#         if not disaster_video:
#             st.warning("Please upload a video to start detection.")

# with tab2:
#     if start_button:
#         if not disaster_video:
#             st.error("Please upload a video file.")
#         else:
#             st.subheader("Detection Results")
            
#             # Process the video and detect objects
#             process_video(disaster_video)
