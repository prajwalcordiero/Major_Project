from fastapi import FastAPI, WebSocket
import asyncio
import numpy as np
import json
import cv2
import os

app = FastAPI()

def generate_heatmap_from_frame(frame):
    # 1. Standardize frame size for consistent math
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Stronger Blur to kill the grass texture
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 3. Canny Edges (Slightly stricter so it only catches hard outlines)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Dilation (Thickens edges)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 5. Resize into the 16x16 structural grid
    resized = cv2.resize(dilated_edges, (16, 16), interpolation=cv2.INTER_AREA)
    heatmap = resized / 255.0
    
    # ---------------------------------------------------------
    # THE FIX: TUNED ADAPTIVE THRESHOLD & OBSERVABILITY
    # ---------------------------------------------------------
    global_density = np.mean(heatmap)
    
    # Print the score to the terminal so you can see the AI's "thought process"
    print(f"Current Global Density Score: {global_density:.3f}")
    
    # We raised the threshold from 0.15 to 0.40!
    if global_density < 0.40:
        # SCENARIO 1: SPARSE CROWD (Aerial Grass Video)
        # Cap the maximum risk at 0.6 so it never triggers the 0.9 Critical alarm.
        heatmap = np.clip(heatmap * 2.5, 0.0, 0.6)
    else:
        # SCENARIO 2: ULTRA-DENSE CROWD (Stadium Video)
        # Allow the multiplier to push hotspots up to 1.0 (Critical Alert).
        heatmap = np.clip(heatmap * 3.5, 0.0, 1.0)
        
    return heatmap.tolist()

@app.websocket("/ws/heatmap")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # =================================================================
    # DEMO CONTROL CENTER: Change this to your video file names
    media_path = "14730886_1920_1080_60fps.mp4" 
    # =================================================================
    
    try:
        is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            cap = cv2.VideoCapture(media_path)
        else:
            frame = cv2.imread(media_path)

        while True:
            # Handle Video Streaming Frame-by-Frame
            if is_video:
                ret, current_frame = cap.read()
                if not ret:
                    # Loop the video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, current_frame = cap.read()
            else:
                current_frame = frame

            # Process the frame
            if current_frame is None:
                print(f"Waiting for {media_path}...")
                density_data = np.zeros((16, 16)).tolist()
            else:
                density_data = generate_heatmap_from_frame(current_frame)
            
            # Fire the data to the React Dashboard
            payload = {
                "type": "HEATMAP_UPDATE",
                "data": density_data,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_text(json.dumps(payload))
            
            # Stream at 10 FPS for video
            await asyncio.sleep(0.1 if is_video else 0.5) 
            
    except Exception as e:
        print(f"Connection closed: {e}")
        if is_video and 'cap' in locals():
            cap.release()