from moviepy.editor import VideoFileClip

def trim_video(input_path, output_path, n_minutes):
    # Convert minutes to seconds
    end_time = n_minutes * 60
    
    # Load the video file
    video = VideoFileClip(input_path)
    
    # Check if the video duration is shorter than the requested end time
    if video.duration < end_time:
        print(f"Video is shorter than {n_minutes} minutes. Trimming to available duration.")
        end_time = video.duration
    
    # Extract the subclip
    subclip = video.subclip(0, end_time)
    
    # Write the trimmed video to the output file
    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Close the clips to free resources
    video.close()
    subclip.close()

import sys
# Example usage
trim_video(sys.argv[1],sys.argv[2], n_minutes=int(sys.argv[3]))
