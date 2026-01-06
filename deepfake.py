import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import librosa
from moviepy.editor import VideoFileClip
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        out, w = self.attn(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        return out, w

class MultiModalFakeDetector(nn.Module):
    def __init__(self, video_dim=512, audio_dim=128, hidden_dim=256, num_classes=3, num_heads=4):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.cross_attn = CrossAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, video_emb, video_mask, audio_emb, audio_mask):
        video_proj = self.video_proj(video_emb)
        audio_proj = self.audio_proj(audio_emb)

        attn_out, attn_weights = self.cross_attn(
            query=video_proj,
            key=audio_proj,
            value=audio_proj,
            key_padding_mask=~audio_mask
        )

        fused = video_proj + attn_out
        fused_pool = F.adaptive_avg_pool1d(fused.permute(0,2,1), 1).squeeze(-1)
        out = self.class_head(fused_pool)
        return out, attn_weights

model = MultiModalFakeDetector().to(device)
model.load_state_dict(torch.load("multimodal_fake_detect - 3 classesor.pth", map_location=device))
model.eval()

# Face Extraction
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def extract_frames(video_path, skip=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if i % skip == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames

def get_face_embeddings(frames):
    embs = []
    for f in frames:
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            e = resnet(face).detach().cpu().numpy()
            embs.append(e)
    if embs:
        return np.vstack(embs)
    return None

# Audio Extraction
def extract_audio_features(video_path, sr=16000, n_mels=128):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is None:
            return None
        audio_path = "temp.wav"
        audio.write_audiofile(audio_path, fps=sr, verbose=False, logger=None)
        y, sr = librosa.load(audio_path, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except:
        return None


# Streamlit
st.title("Multimodal Deepfake Detector")
st.write("Upload a video to analyze fake/genuine cues from audio–video attention.")

uploaded = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded.read())

    st.video("temp_video.mp4")

    with st.spinner("Extracting frames..."):
        frames = extract_frames("temp_video.mp4")

    if len(frames) == 0:
        st.error("Could not read video frames.")
        st.stop()

    with st.spinner("Computing face embeddings..."):
        video_emb = get_face_embeddings(frames)

    if video_emb is None:
        st.error("No face detected in video.")
        st.stop()

    with st.spinner("Extracting audio features..."):
        audio_emb = extract_audio_features("temp_video.mp4")

    if audio_emb is None:
        st.warning("Audio not detected. Using zeros.")
        audio_emb = np.zeros((128, 1))

    # Convert to tensors
    video_emb = torch.tensor(video_emb, dtype=torch.float32)
    audio_emb = torch.tensor(audio_emb.T, dtype=torch.float32)

    # Padding (single-sample batch)
    videos = pad_sequence([video_emb], batch_first=True)
    audios = pad_sequence([audio_emb], batch_first=True)

    video_mask = torch.tensor([[i < video_emb.shape[0] for i in range(videos.size(1))]])
    audio_mask = torch.tensor([[i < audio_emb.shape[0] for i in range(audios.size(1))]])

    videos = videos.to(device)
    audios = audios.to(device)
    video_mask = video_mask.to(device)
    audio_mask = audio_mask.to(device)

    with st.spinner("Running model..."):
        logits, attn = model(videos, video_mask, audios, audio_mask)
        pred = torch.argmax(logits, dim=1).item()
        attn = attn[0].mean(dim=0).detach().cpu().numpy()


    # Show Prediction
    st.subheader("Prediction")
    label_map = {0: "FakeVideo-RealAudio", 1: "RealVideo-FakeAudio", 2: "FakeVideo-FakeAudio"}
    st.write(f"**Predicted Class:** {label_map[pred]}")

    # Show Attention Heatmap
    st.subheader("Cross-Attention Heatmap (Video → Audio)")

    with torch.no_grad():
        # Same transformations the model applies internally
        video_proj = model.video_proj(videos)      # (1, Tv, hidden_dim)
        audio_proj = model.audio_proj(audios)      # (1, Ta, hidden_dim)

        # Remove batch dimension
        video_proj = video_proj[0]   # (Tv, hidden_dim)
        audio_proj = audio_proj[0]   # (Ta, hidden_dim)

    # Compute cosine similarity matrix
    similarity_map = F.cosine_similarity(
        video_proj.unsqueeze(1),    # (Tv, 1, hidden_dim)
        audio_proj.unsqueeze(0),    # (1, Ta, hidden_dim)
        dim=-1
    ).cpu()

    # Heatmap ----
    plt.figure(figsize=(11, 6))
    sns.heatmap(
        similarity_map,
        cmap="coolwarm",
        center=0,
        xticklabels=5,
        yticklabels=5
    )

    plt.xlabel("Audio Timesteps")
    plt.ylabel("Video Frames")
    plt.title(f"Audio–Video Cosine Similarity, Label={label_map[pred]})")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
