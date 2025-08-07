import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM 
from crewai_tools import SerperDevTool

from dotenv import load_dotenv

load_dotenv()

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class BusinessPlan():
	"""BusinessPlan crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def market_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['market_researcher'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@agent 
	def business_strategist(self) -> Agent:
		return Agent(
			config=self.agents_config['business_strategist'],
			verbose=True
		)

	@agent
	def marketing_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['marketing_expert'],
			verbose=True
		)

	@agent
	def financial_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['financial_analyst'],
			verbose=True
		)
	@agent
	def competitor_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['competitor_analyst'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@task
	def competitor_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_analysis_task']
		)

	@agent
	def operations_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['operations_specialist'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def market_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['market_research_task']
		)

	@task
	def business_strategy_task(self) -> Task:
		return Task(
			config=self.tasks_config['business_strategy_task']
		)

	@task
	def marketing_plan_task(self) -> Task:
		return Task(
			config=self.tasks_config['marketing_plan_task']
		)

	@task
	def financial_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['financial_analysis_task']
		)
	@task
	def competitor_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['competitor_analysis_task']
		)

	@task
	def operations_plan_task(self) -> Task:
		return Task(
			config=self.tasks_config['operations_plan_task']
		)

	@task
	def final_report_task(self) -> Task:
		"""Task to combine all outputs and write to a markdown file."""
		return Task(
			description="Combine all previous task outputs into a final business plan report in markdown format and save it to a file.",
			expected_output="A markdown file containing the complete business plan.",
			callback=self.write_report_to_file,
		)

	def write_report_to_file(self, task_output, **kwargs):
		"""Callback function to write the final report to a markdown file."""
		if not task_output:
			print("No output to write to file.")
			return

		file_path = "business_plan.md"
		try:
			with open(file_path, "w", encoding="utf-8") as file:
				file.write(str(task_output))
			print(f"Business plan written to {file_path}")
		except Exception as e:
			print(f"Error writing to file: {e}")

	@crew
	def crew(self) -> Crew:
		"""Creates the BusinessPlan crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			manager_llm=LLM(
				model="gemini/gemini-2.0-flash-exp",
				api_key=os.getenv("GEMINI_API_KEY"),
				system_message="You are an expert project manager. Your job is to ensure that the crew is running smoothly and that the business plan is completed on time and effectively."
			),
			planning=True,
			verbose=True,
			max_rpm=10,
			process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
