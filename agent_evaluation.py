import os, json
import pandas as pd
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from user_functions import user_functions
from dotenv import load_dotenv


openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

load_dotenv()

from azure.ai.projects import __version__ as projects_version
from packaging.version import Version
# some dependencies have been updated with breaking changes -- indicates whether to use updated models and APIs or not
updated_agents = Version(projects_version) > Version("1.0.0b10") or projects_version.startswith("1.0.0a")

if updated_agents:
    from azure.ai.agents.models import FunctionTool, ToolSet
    project_client = AIProjectClient(
        endpoint="https://ayushija-2422-resource.services.ai.azure.com/api/projects/ayushija-2422",
        credential=DefaultAzureCredential()
    )
else:
    from azure.ai.projects.models import FunctionTool, ToolSet
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ["PROJECT_CONNECTION_STRING"],
    )

AGENT_NAME = "Seattle Tourist Assistant PrP"

# Adding Tools to be used by Agent 
functions = FunctionTool(user_functions)

toolset = ToolSet()
toolset.add(functions)

agent = project_client.agents.create_agent(
    model=deployment,
    name="my-assistant",
    instructions="You are a helpful assistant",
    toolset=toolset
)

agent = project_client.agents.get_agent(
    agent_id = "asst_aAgU5FeUSgDY2WhIG7KrZxZs"
)

print(f"Fetched agent, ID: {agent.id}")

if updated_agents:
    thread = project_client.agents.threads.create()
else:
    thread = project_client.agents.create_thread()
print(f"Created thread, ID: {thread.id}")

# Create message to thread

MESSAGE = "Can you send me an email with weather information for Seattle?"

if updated_agents:
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=MESSAGE,
    )        
else:
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=MESSAGE,
    )
    
print(f"Created message, ID: {message.id}")

if updated_agents:
    from azure.ai.agents.models import (
        FunctionTool,
        ListSortOrder,
        RequiredFunctionToolCall,
        SubmitToolOutputsAction,
        ToolOutput,
    )
    run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent.id)
    
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            if not tool_calls:
                print("No tool calls provided - cancelling run")
                project_client.agents.runs.cancel(thread_id=thread.id, run_id=run.id)
                break

            tool_outputs = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        print(f"Executing tool call: {tool_call}")
                        output = functions.execute(tool_call)
                        tool_outputs.append(
                            ToolOutput(
                                tool_call_id=tool_call.id,
                                output=output,
                            )
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            print(f"Tool outputs: {tool_outputs}")
            if tool_outputs:
                project_client.agents.runs.submit_tool_outputs(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
    print(f"Run status: {run.status}")

else:
    from azure.ai.projects.models import (
        FunctionTool,
        ListSortOrder,
        RequiredFunctionToolCall,
        SubmitToolOutputsAction,
        ToolOutput,
    )
    run = project_client.agents.create_run(thread_id=thread.id, agent_id=agent.id)
    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            if not tool_calls:
                print("No tool calls provided - cancelling run")
                project_client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
                break

            tool_outputs = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        print(f"Executing tool call: {tool_call}")
                        output = functions.execute(tool_call)
                        tool_outputs.append(
                            ToolOutput(
                                tool_call_id=tool_call.id,
                                output=output,
                            )
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            print(f"Tool outputs: {tool_outputs}")
            if tool_outputs:
                project_client.agents.submit_tool_outputs_to_run(thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs)
    print(f"Run status: {run.status}")


print(f"Run finished with status: {run.status}")

if run.status == "failed":
    print(f"Run failed: {run.last_error}")

print(f"Run ID: {run.id}")


if updated_agents:
    for message in project_client.agents.messages.list(thread.id, order="asc"):
        print(f"Role: {message.role}")
        print(f"Content: {message.content[0].text.value}")
        print("-" * 40)
else:
    for message in project_client.agents.list_messages(thread.id, order="asc").data:
        print(f"Role: {message.role}")
        print(f"Content: {message.content[0].text.value}")
        print("-" * 40)

import json
from azure.ai.evaluation import AIAgentConverter

# Initialize the converter that will be backed by the project.
converter = AIAgentConverter(project_client)

thread_id = thread.id
run_id = run.id

converted_data = converter.convert(thread_id=thread_id, run_id=run_id)
print(json.dumps(converted_data, indent=4))
# Save the converted data to a JSONL file

file_name = "evaluation_data.jsonl"
evaluation_data = converter.prepare_evaluation_data(thread_ids=thread.id, filename=file_name)

load_dotenv()
from azure.ai.evaluation import ToolCallAccuracyEvaluator , AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator, ViolenceEvaluator
from pprint import pprint

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=openai_endpoint,
    api_key=openai_key,
    api_version=api_version,
    azure_deployment=deployment,
)
# Needed to use content safety evaluators
azure_ai_project={
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
    "project_name": "ayushija-dummy-resource",
    "resource_group_name": "rg-ayushija-2422",
}

tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
intent_resolution = IntentResolutionEvaluator(model_config=model_config)
task_adherence = TaskAdherenceEvaluator(model_config=model_config)


tool_call_accuracy(query=converted_data['query'], response=converted_data['response'], tool_definitions=converted_data['tool_definitions'])


from azure.ai.evaluation import evaluate

response = evaluate(
    data=file_name,
    evaluators={
        "tool_call_accuracy": tool_call_accuracy,
        "intent_resolution": intent_resolution,
        "task_adherence": task_adherence,
    },
    azure_ai_project={
        "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
      "project_name": "ayushija-dummy-resource",
      "resource_group_name": "rg-ayushija-2422",
    }
)
pprint(f'AI Foundary URL: {response.get("studio_url")}')
pprint(response)

