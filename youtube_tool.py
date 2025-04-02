from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import re

class YouTubeTranscriptInput(BaseModel):
    """Input schema for YouTubeTranscriptTool."""
    video_url: str = Field(..., description="YouTube video URL or ID to extract transcript from.")

class YouTubeTranscriptTool(BaseTool):
    name: str = "YouTube Transcript Tool"
    description: str = "Fetches transcript from a YouTube video using its URL or ID."
    args_schema: Type[BaseModel] = YouTubeTranscriptInput

    def _extract_video_id(self, url: str) -> str:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        return match.group(1) if match else url.strip()

    def _run(self, video_url: str) -> str:
        try:
            video_id = self._extract_video_id(video_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry["text"] for entry in transcript])
            return full_text[:4000]  # Return up to 4000 characters for context fitting
        except Exception as e:
            return f"⚠️ Failed to fetch transcript: {str(e)}"
