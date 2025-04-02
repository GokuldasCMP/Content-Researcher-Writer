from crewai import Agent, Task, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import agentops
from orch_two import ModuleOrchestrator
from youtube_tool import YouTubeTranscriptTool  # Import your custom tool

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv("SERPER_API_KEY")

agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

# Define the module/topic (you can dynamically change this)
topic = "Statistics In DataScience"

# Tools: Web search + YouTube transcript fetcher
search_tool = SerperDevTool(n=10)
yt_tool = YouTubeTranscriptTool()

# LLM to be used across agents
llm = LLM(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# Agent 1: Content Gatherer that extracts *real* transcripts
content_gatherer = Agent(
    role="YouTube Transcript Collector",
    goal=f"Search YouTube videos on '{topic}' and extract real spoken transcripts using YouTube Transcript Tool.",
    backstory=(
        "You're a smart content extractor who ONLY deals with YouTube video URLs. "
        "You use a transcript tool to fetch *actual spoken words* from each video, not just search snippets. "
        "Your job is to deliver raw but accurate spoken content from the videos, ready for refinement."
    ),
    tools=[search_tool, yt_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Agent 2: Transcript Refiner
contextual_refiner = Agent(
    role="Transcript Refiner",
    goal="Transform YouTube transcripts into structured, digestible learning material.",
    backstory=(
        "You take raw transcripts from video content and refactor them into polished, beginner-friendly educational modules. "
        "You're excellent at summarizing, segmenting, and clarifying spoken content for structured learning."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Task 1: Gather full transcripts from YouTube videos
gather_task = Task(
    description=(
        f"""
        Use web search to find YouTube videos about "{topic}".
        For each video:
        - Extract video title and URL
        - Use the YouTube Transcript Tool to fetch the full spoken transcript
        - Do NOT invent or summarize. Only use the actual transcript content.

        Format:
        1. Title
        2. Video URL
        3. Full Transcript
        """
    ),
    expected_output=(
        "A list of 3–5 YouTube videos with:\n"
        "- Video title\n"
        "- Video URL\n"
        "- Full transcript (up to 4000 characters)"
    ),
    agent=content_gatherer
)

# Task 2: Refine transcripts into learning material
refine_task = Task(
    description=(
        f"""
        Refine the gathered YouTube transcripts about "{topic}" into structured educational content.

        Keep:
        - Definitions, examples, concepts, formulas mentioned in the transcript

        Clean:
        - Filler words, repetition, irrelevant or casual language

        Format:
        - Headings: Definitions, Core Concepts, Formulas, Examples
        - Use bullet points or short paragraphs
        - Maintain factual tone. DO NOT hallucinate content.
        """
    ),
    expected_output=(
        "Refined content including:\n"
        "- Clearly structured sections\n"
        "- Beginner-friendly tone\n"
        "- Quotes or rephrased transcript lines grouped into learning themes"
    ),
    agent=contextual_refiner
)

# Orchestrator with only 2 tasks/orchestrated flow
orchestrator = ModuleOrchestrator(
    gather_task=gather_task,
    refine_task=refine_task,
    topic=topic
)

# Run it
final_result = orchestrator.run_pipeline()

# Show final result
print("\n\u2705 FINAL OUTPUT:\n", final_result)
print("\n\U0001f9e0 MEMORY SNAPSHOT:\n", orchestrator.memory.get_history())