from google import genai
from google.genai import types

# Only for videos of size <20Mb
video_file_name = "./media/video-audio.mp4"
video_bytes = open(video_file_name, 'rb').read()

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
            ),
            types.Part(text='Does the video contains audio? If yes, please try to transcribe the audio.')
        ]
    )
)
print(response.text)