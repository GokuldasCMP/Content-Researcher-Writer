from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import agentops
from orch_memory import ModuleOrchestrator

load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv("SERPER_API_KEY")

agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))

# Define the module/topic (you can dynamically change this)
topic = "Statistics In DataScience"

# Tool: Web Search Tool for Content Gathering
search_tool = SerperDevTool(n=10)

# LLM to be used across agents
llm = LLM(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

content_gatherer = Agent(
    role="Content Gatherer",
    goal=f"Pull diverse structured and unstructured content on the topic: {topic}",
    backstory=(
        "You're an expert content miner specialized in gathering both structured and unstructured data "
        "from reliable sources like blogs, YouTube transcripts, PDFs, forums, and documentation. "
        "You prioritize diverse sources and extract relevant insights, examples, and terminology."
    ),
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

contextual_refiner = Agent(
    role="Contextual Refiner",
    goal="Filter, clean, and align content with internal knowledge style and structure",
    backstory=(
        "You're a skilled content editor with deep understanding of your organization's knowledge standards. "
        "You filter noisy or irrelevant parts, remove redundancies, align content tone, and rewrite segments to match internal voice and clarity."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

output_composer = Agent(
    role="Structured Output Composer",
    goal="Convert refined content into a structured format with well-defined topics and subtopics",
    backstory=(
        "You're a content architect specializing in transforming raw insights into a structured format "
        "used by data science learners. You categorize information into topics, subtopics, examples, key takeaways, "
        "and organize them according to a pre-approved Excel or web-based outline."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

quality_validator = Agent(
    role="Content Quality Validator",
    goal="Ensure content is clean, coherent, non-redundant, and high quality",
    backstory=(
        "You're a seasoned content auditor responsible for final-stage quality checks. "
        "You review clarity, tone consistency, factual accuracy, continuity, and remove any duplication. "
        "You ensure the output is aligned with pedagogical standards and ready for deployment."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

evaluation_agent = Agent(
    role="Content Evaluator",
    goal="Evaluate the final output quality based on content standards and a predefined rubric",
    backstory=(
        "You're a meticulous evaluation specialist responsible for ensuring learning modules meet high educational standards. "
        "You assess relevance, completeness, tone, and factual alignment with the topic. You provide a score and brief reasoning."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)


# Task 1: Content Gathering
gather_task = Task(
    description=(
        f"""
        Gather high-quality structured and unstructured content on the topic: "{topic}" from the web.
        Include content from:
        - Blogs
        - YouTube transcripts (if available)
        - PDFs and academic sources
        - Forums and documentation (like Stack Overflow, official docs)
        Extract raw content, examples, definitions, and use cases. Include source references.
        """
    ),
    expected_output=(
        "A raw content dump organized by type (blog, video, docs), with key points, examples, and source URLs."
    ),
    agent=content_gatherer
)

# Task 2: Contextual Refining
refine_task = Task(
    description=(
        """
        Given the gathered content on the topic "{topic}", refine it by:
        - Removing redundant or irrelevant parts
        - Summarizing verbose text
        - Retaining important "{topic}" examples, use cases, definitions
        - Rewriting in a tone aligned to beginner/intermediate learners

        🔒 Only use the provided content. Do NOT invent unrelated examples.
        
        🧠 Example:
        Original: "In this advanced SQL lecture, we'll explore nested queries and their complexities..."
        Rewritten: "We’ll cover nested SQL queries and how to use them for real-world data filtering tasks in analytics."
        """ 
    ),
    expected_output=(
        "Cleaned and well-aligned learning content broken into paragraphs, bullet points, and topic-related examples."
    ),
    agent=contextual_refiner
)


# Task 3: Structuring Output
compose_task = Task(
    description=(
        """
        Using the refined content and the topic "{topic}", structure a learning module with the following format:
        - Overview
        - Topics & Subtopics
        - Key Concepts
        - Practical Examples (using "{topic}")
        - Summary Notes
        - Source Links
        
        Make sure the content is only about "{topic}" in Data Science. Avoid introducing unrelated topics like  generic learning advice.
        """
    ),
    expected_output=(
        "Markdown-structured or JSON output organized with section headers matching internal learning module format."
    ),
    agent=output_composer
)


# Task 4: Final Validation
validate_task = Task(
    description=(
        """
        Review the final content for the topic "{topic}". Your task is to:
        - Ensure tone, formatting, and structure match internal learning material
        - Check factual accuracy and "{topic}" terminology
        - Remove any hallucinated, irrelevant, or unrelated parts
        - Avoid generic or copy-pasted placeholder content
        
        Do not introduce anything beyond the given topic scope.
        """
    ),
    expected_output=(
        "Polished, error-free learning module content focused only on the assigned topic, ready to publish."
    ),
    agent=quality_validator
)

evaluation_task = Task(
    description=(
        f"""
        Evaluate the final module output for the topic "{topic}" based on the following:
        - Relevance to topic
        - Completeness of explanation
        - Clarity and beginner-friendliness
        - Correct use of terminology
        - Structural organization (headings, subpoints, examples)

        Provide a final rating out of 10 and a short paragraph justifying the rating.
        """
    ),
    expected_output="Score out of 10 with a paragraph explaining strengths and weaknesses of the content.",
    agent=evaluation_agent
)


# Instantiate the orchestrator
orchestrator = ModuleOrchestrator(
    gather_task=gather_task,
    refine_task=refine_task,
    compose_task=compose_task,
    validate_task=validate_task,
    evaluation_task=evaluation_task,
    topic=topic
)

# Run the orchestrated pipeline
final_result = orchestrator.run_pipeline()

# Print the final structured output
print("\n✅ FINAL OUTPUT:\n", final_result)
print("\n🧠 MEMORY SNAPSHOT:\n", orchestrator.memory.get_history())


