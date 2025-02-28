# test_download.py
import os
import yt_dlp as youtube_dl
import sys

# Define the download function (if it's not already imported from your code)
def download_youtube_audio(youtube_url, output_path):
    print("Starting download for:", youtube_url)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print("Audio downloaded to:", output_path)
    except Exception as e:
        print("Error during download:", e)
        sys.exit(1)

if __name__ == "__main__":
    # Replace with your full-length video URL
    test_url = "https://youtu.be/Gztabfs5ngA?si=i3kvioFnqds2TwGe"
    # Make sure the output file path includes a filename (e.g., full_track.wav)
    output_file = os.path.join("C:/Users/goldw/PycharmProjects/DrumTracksAI/downloaded_audio", "full_track")
    download_youtube_audio(test_url, output_file)
