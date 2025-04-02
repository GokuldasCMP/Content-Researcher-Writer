import time
import traceback
from crewai import Crew
from memory_layer import MemoryLayer  # 👈 Add memory layer import

class ModuleOrchestrator:
    def __init__(self, gather_task, refine_task, topic):
        self.gather_task = gather_task
        self.refine_task = refine_task
        self.topic = topic
        self.logs = []
        self.memory = MemoryLayer()  # 👈 Initialize memory layer

    def log(self, step, status, detail=""):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "detail": detail
        }
        self.logs.append(log_entry)
        print(f"[{timestamp}] [{step}] [{status}] {detail}")

    def execute_task(self, task, input_data, retries=2):
        for attempt in range(retries):
            try:
                self.log(task.agent.role, f"Attempt {attempt + 1}", "Running task...")

                # Inject topic to input_data
                input_data["topic"] = self.topic

                # Inject memory into input
                input_data = self.memory.inject_memory(input_data)

                self.log(task.agent.role, "Input", str(input_data))
                start_time = time.time()

                mini_crew = Crew(
                    agents=[task.agent],
                    tasks=[task],
                    verbose=False
                )
                result = mini_crew.kickoff(inputs=input_data)

                duration = time.time() - start_time
                self.log(task.agent.role, "Timing", f"⏱️ Took {duration:.2f} seconds")

                # Normalize the result
                if hasattr(result, "output"):
                    result_output = result.output
                elif isinstance(result, dict) and "output" in result:
                    result_output = result["output"]
                else:
                    result_output = str(result)

                self.log(task.agent.role, "Output", str(result_output))

                if isinstance(result_output, str) and self.topic.lower() not in result_output.lower():
                    self.log(task.agent.role, "Warning", "⚠️ Output may be unrelated to the topic.")

                if isinstance(result_output, str) and len(result_output.strip()) > 50:
                    self.log(task.agent.role, "Success")
                    self.memory.remember(task.agent.role, result_output)
                    return result_output
                else:
                    self.log(task.agent.role, "Warning", f"Received output type: {type(result_output)}")
                    raise ValueError("Empty or insufficient output.")

            except Exception as e:
                self.log(task.agent.role, "Error", str(e))
                self.log(task.agent.role, "Traceback", traceback.format_exc())
                time.sleep(1)

        self.log(task.agent.role, "Failed", "Max retries reached.")
        return None

    def run_pipeline(self):
        self.log("Orchestrator", "Starting", f"Generating content for topic: {self.topic}")

        # Step 1: Gather Content
        raw_content = self.execute_task(self.gather_task, {"topic": self.topic})
        if not raw_content:
            return "❌ Pipeline failed at Content Gathering."

        # Step 2: Refine Content
        refined = self.execute_task(self.refine_task, {"gathered_content": raw_content, "topic": self.topic})
        if not refined:
            return "❌ Pipeline failed at Contextual Refining."

        self.log("Orchestrator", "Completed", "Module content created successfully.")
        return refined

    def get_logs(self):
        return self.logs
