import base64
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

video_bytes = open("./media/video-audio.mp4", "rb").read()
video_base64 = base64.b64encode(video_bytes).decode("utf-8")
mime_type = "video/mp4"

message = HumanMessage(
    content=[
        {"type": "text", "text": "Does the video contains audio? If yes, please try to transcribe the audio. You should contain timestamp in your response."},
        {
            "type": "video",
            "base64": video_base64,
            "mime_type": mime_type,
        },
    ]
)
response = model.invoke([message])
print(response.content)

# google原生api和langchain api的对比，均可以处理带音频的视频。并且都可以带时间戳。