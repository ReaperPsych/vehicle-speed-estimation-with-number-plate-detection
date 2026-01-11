import os
import json
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required 
from .detector import VehicleDetector

# @login_required
def dashboard(request):
    context = {}
    fs = FileSystemStorage()

    # --- RESET LOGIC ---
    # Clearly wipe session data if user requests a reset
    if request.method == 'GET' and request.GET.get('action') == 'reset':
        if 'uploaded_filename' in request.session:
            # Optional: Delete the file from disk when resetting
            old_file = request.session['uploaded_filename']
            if fs.exists(old_file):
                fs.delete(old_file)
            del request.session['uploaded_filename']
            
        if 'uploaded_file_url' in request.session:
            del request.session['uploaded_file_url']
            
        return redirect('dashboard')

    if request.method == 'POST':
        if 'video_file' in request.FILES:
            # --- STEP 1: VIDEO UPLOAD ---
            try:
                # CLEANUP: If there's already a file in the session (abandoned upload), remove it
                if 'uploaded_filename' in request.session:
                    old_file = request.session['uploaded_filename']
                    if fs.exists(old_file):
                        fs.delete(old_file)
                
                # Save new file
                video_file = request.FILES['video_file']
                filename = fs.save(video_file.name, video_file)
                
                # Update Session
                request.session['uploaded_filename'] = filename
                request.session['uploaded_file_url'] = fs.url(filename)
                
                context['uploaded_file_url'] = fs.url(filename)
                context['show_calibration'] = True 
                context['message'] = "Video uploaded. Please configure calibration settings below."
                
            except Exception as e:
                context['error'] = f"Error during upload: {str(e)}"
                
        elif 'calibration_points' in request.POST:
            # --- STEP 2: CALIBRATION & PROCESSING ---
            filename = request.session.get('uploaded_filename')
            if not filename:
                context['error'] = "Session expired or file missing. Please upload again."
                return render(request, 'dashboard.html', context)

            try:
                # 1. Get Calibration Points
                points_json = request.POST.get('calibration_points')
                source_points_list = json.loads(points_json) 
                
                # 2. Get User Inputs for Real World Data & Model
                real_width = float(request.POST.get('zone_width', 4.0))
                real_height = float(request.POST.get('zone_length', 10.0))
                model_choice = request.POST.get('model_type', 'yolov8n.pt') 
                
                # Paths
                base_dir = settings.MEDIA_ROOT
                source_path = os.path.join(base_dir, filename)
                output_filename = f"processed_{filename}"
                output_path = os.path.join(base_dir, output_filename)
                
                # 3. Initialize Detector
                print(f"Initializing Detector with {model_choice}...")
                detector = VehicleDetector(model_path=model_choice) 
                
                print(f"Starting processing (Zone: {real_width}m x {real_height}m)...")
                
                # 4. Run Processing
                results = detector.process_video(
                    source_path, 
                    output_path, 
                    source_points_list,
                    real_width,
                    real_height
                )
                
                # 5. Pass results
                context['uploaded_file_url'] = request.session.get('uploaded_file_url') 
                context['processed_file_url'] = fs.url(output_filename)
                context['total_vehicles'] = results['total_vehicles']
                context['avg_speed'] = results['avg_speed']
                context['max_speed'] = results['max_speed']
                context['traffic_flow_level'] = results['traffic_flow_level']
                context['traffic_flow_color'] = results['traffic_flow_color']
                context['message'] = "Analysis Complete!"
                
                # Clear session data (Analysis done)
                if 'uploaded_filename' in request.session: del request.session['uploaded_filename']
                if 'uploaded_file_url' in request.session: del request.session['uploaded_file_url']

            except Exception as e:
                print(f"Error during processing: {e}")
                context['error'] = f"Error during processing: {str(e)}"
    
    # Handle page reload during calibration step (Persistent State)
    elif request.session.get('uploaded_filename') and not request.POST:
        context['uploaded_file_url'] = request.session.get('uploaded_file_url')
        context['show_calibration'] = True
        context['message'] = "Resume calibration or upload a new video."

    return render(request, 'dashboard.html', context)