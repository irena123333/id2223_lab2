import gradio as gr
from transformers import pipeline
from pytube import YouTube

pipe = pipeline(model="irena/whisper-small-sv-SE")

def transcribe_video(url):  
  yt=YouTube(url).streams.filter(only_audio=True).all()
  audio=yt[0].download()
  text = pipe(audio)["text"]
  return text

def transcribe_audio(audio):
  text = pipe(audio)["text"]
  return text



audio = gr.Interface(
    fn=transcribe_audio, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)



video = gr.Interface(
	fn=transcribe_video,
	inputs=gr.Textbox(label="Enter a YouTube URL:"),
	outputs="text",
	title="Whisper Small Swedish",
	description="Transcribe swedish videos from YouTube",
)




demo = gr.TabbedInterface([audio, video], ["transcribe from recording", "transcribe from youtube url"])

if __name__ == "__main__":
    demo.launch()

 











