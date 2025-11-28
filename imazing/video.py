
import cv2
import numpy as np
import time
import os
from .core import Imazing
from .utils import ensure_valid, check_dependency

class VideoManager:
    """
    Handles Video Processing, Editing, and Biometric Analysis.
    """
    
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Warning: Could not open video source: {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0: self.fps = 30.0

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    # --- CORE PROCESSOR ---
    
    def process_stream(self, processor_func, display=True, write_path=None):
        """
        The Main Loop.
        processor_func: callback(Imazing_obj) -> returns modified Imazing_obj
        """
        out = None
        if write_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(write_path, fourcc, self.fps, (self.width, self.height))
            
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            img_obj = Imazing(frame)
            try:
                # Run the logic
                processed_obj = processor_func(img_obj)
                final_frame = processed_obj.image
            except Exception as e:
                print(f"Processing Error: {e}")
                final_frame = frame
                
            if out: out.write(final_frame)
            
            if display:
                cv2.imshow("Imazing Stream (Press 'q' to quit)", final_frame)
                # Quit logic
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        if out: out.release()
        cv2.destroyAllWindows()

    # --- BIOMETRICS (Fixed: Visual + Interactive) ---

    def detect_motion(self, threshold=3200, alpha=0.1):
        """
        High-Sensitivity Motion Detection.
        Shows the feed. Press 'q' to stop monitoring and return timestamps.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        motion_timestamps = []
        background = None
        last_motion_time = 0
        cooldown = 2.0
        
        print("📹 Motion Monitor Active. Press 'q' to stop.")

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            # 1. Processing (Gray + Blur)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # 2. Smart Background Adaptation
            if background is None:
                background = gray.copy().astype("float")
                continue
            
            cv2.accumulateWeighted(gray, background, alpha)
            
            # 3. Difference
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(background))
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # 4. Contour Check
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_found = False
            
            for c in cnts:
                if cv2.contourArea(c) > threshold:
                    motion_found = True
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 5. UI & State
            current_time = time.time()
            if motion_found:
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                motion_timestamps.append(timestamp)
                last_motion_time = current_time
            
            if (current_time - last_motion_time) < cooldown:
                cv2.putText(frame, "WARNING: INTRUDER", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Status: Safe", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Motion Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return motion_timestamps

    def verify_liveness_smile(self, timeout=10):
        """
        Interactive Smile Check. Shows camera feed.
        """
        start_time = time.time()
        face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        print("Please Smile for the Camera...")
        
        while (time.time() - start_time) < timeout:
            ret, frame = self.cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Draw Face Box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)
                
                if len(smiles) > 0:
                    cv2.putText(frame, "SMILE DETECTED!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Liveness Check", frame)
                    cv2.waitKey(1000) # Pause to show success
                    cv2.destroyAllWindows()
                    return True
            
            # Show timer
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"Time: {timeout - elapsed}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Liveness Check (Smile)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return False

    def verify_liveness_blink(self, timeout=10):
        """
        Interactive Blink Check. Shows camera feed.
        """
        start_time = time.time()
        face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        eyes_visible_prev = False
        blink_detected = False
        
        print("Please Blink...")
        
        while (time.time() - start_time) < timeout:
            ret, frame = self.cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cas.detectMultiScale(gray, 1.3, 5)
            
            eyes_now = False
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cas.detectMultiScale(roi_gray, 1.1, 10)
                
                # Draw eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
                
                if len(eyes) >= 2: eyes_now = True
            
            # Logic
            if eyes_visible_prev and not eyes_now:
                pass # Eyes closed
            elif not eyes_visible_prev and eyes_now and blink_detected:
                cv2.putText(frame, "BLINK VERIFIED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Liveness Check (Blink)", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                return True
                
            if eyes_visible_prev and not eyes_now:
                blink_detected = True
                
            eyes_visible_prev = eyes_now
            
            # Instructions
            cv2.putText(frame, "Look at camera and BLINK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Liveness Check (Blink)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cv2.destroyAllWindows()
        return False

    def verify_head_turn(self, direction='left', timeout=5):
        start = time.time()
        face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"Please turn head {direction}...")
        
        while (time.time() - start) < timeout:
            ret, frame = self.cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cas.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                cv2.putText(frame, "Face Detected - NOW TURN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                for (x,y,w,h) in faces:
                     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            else:
                # Face disappeared (simulated turn)
                cv2.putText(frame, "TURN DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Head Turn Check", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
                return True
                
            cv2.imshow("Head Turn Check", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cv2.destroyAllWindows()
        return False

    # --- VIDEO EDITING ---

    def trim(self, start_sec, end_sec, output_path):
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        current_frame = start_frame
        while current_frame < end_frame and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            out.write(frame)
            current_frame += 1
        out.release()
        return True

    def extract_frames(self, output_folder, every_n_frame=30):
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count = 0
        saved_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            if count % every_n_frame == 0:
                fname = os.path.join(output_folder, f"frame_{count:05d}.jpg")
                cv2.imwrite(fname, frame)
                saved_count += 1
            count += 1
        return saved_count

    def resize_video(self, width, height, output_path):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            resized = cv2.resize(frame, (width, height))
            out.write(resized)
        out.release()

    def to_gif(self, start_sec, duration_sec, output_path, fps=10, resize_width=320):
        Image = check_dependency("PIL.Image")
        if not Image: return
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * self.fps))
        max_frames = int(duration_sec * self.fps)
        step = int(self.fps / fps)
        count = 0
        while count < max_frames:
            ret, frame = self.cap.read()
            if not ret: break
            if count % step == 0:
                h, w = frame.shape[:2]
                r = resize_width / float(w)
                dim = (resize_width, int(h * r))
                resized = cv2.resize(frame, dim)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            count += 1
        if frames:
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)

    def estimate_pulse_basic(self, duration=10):
        start = time.time()
        signals = []
        print("Measuring Pulse (Stay still)...")
        while (time.time() - start) < duration:
            ret, frame = self.cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            # ROI Forehead
            roi = frame[int(h*0.2):int(h*0.3), int(w*0.4):int(w*0.6)]
            cv2.rectangle(frame, (int(w*0.4), int(h*0.2)), (int(w*0.6), int(h*0.3)), (0,255,0), 2)
            cv2.imshow("Pulse Check", frame)
            signals.append(np.mean(roi[:, :, 1]))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        return signals

    def verify_filter_resistance(self):
        ret, frame = self.cap.read()
        if not ret: return False
        face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(face_cas.detectMultiScale(gray, 1.1, 4)) == 0: return False
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        return len(face_cas.detectMultiScale(blur, 1.1, 4)) > 0

    def add_dynamic_watermark(self, output_path, text_prefix="REC"):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        print("Adding Watermark... (Press q to cancel view, but process continues)")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, f"{text_prefix} | {ts}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
            # Optional preview if it's a webcam source
            if self.source == 0:
                cv2.imshow("Watermarking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        out.release()
        cv2.destroyAllWindows()
        
    def scramble_frame(self, rows=2, cols=2):
        ret, frame = self.cap.read()
        if not ret: return None, None
        h, w = frame.shape[:2]
        chunk_h, chunk_w = h // rows, w // cols
        chunks = []
        for r in range(rows):
            for c in range(cols):
                y, x = r * chunk_h, c * chunk_w
                chunks.append(frame[y:y+chunk_h, x:x+chunk_w])
        indices = list(range(len(chunks)))
        np.random.shuffle(indices)
        shuffled_chunks = [chunks[i] for i in indices]
        grid_rows = []
        for r in range(rows):
            row_chunks = shuffled_chunks[r*cols : (r+1)*cols]
            grid_rows.append(np.hstack(row_chunks))
        return np.vstack(grid_rows), indices

    def segment_video(self, output_folder, segment_seconds=10):
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames_per_seg = int(self.fps * segment_seconds)
        seg_idx = 0
        while True:
            out_path = os.path.join(output_folder, f"segment_{seg_idx:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, self.fps, (self.width, self.height))
            frames_written = 0
            while frames_written < frames_per_seg:
                ret, frame = self.cap.read()
                if not ret: break
                out.write(frame)
                frames_written += 1
            out.release()
            seg_idx += 1
            if not ret: break
            
    @staticmethod
    def write_video_from_frames(frames, output_path, fps=30):
        if not frames: return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
        out.release()