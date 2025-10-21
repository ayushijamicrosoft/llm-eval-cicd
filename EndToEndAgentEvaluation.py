from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario, DirectAttackSimulator, IndirectAttackSimulator
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.evaluation.simulator import Simulator
from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import os
from openai import AzureOpenAI
import importlib.resources as pkg_resources
from dotenv import load_dotenv
import os, json
import pandas as pd
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import pprint
from azure.ai.evaluation import ToolCallAccuracyEvaluator , AzureOpenAIModelConfiguration, IntentResolutionEvaluator, TaskAdherenceEvaluator,RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, ViolenceEvaluator, SelfHarmEvaluator, SexualEvaluator, HateUnfairnessEvaluator, CodeVulnerabilityEvaluator, IndirectAttackEvaluator, ProtectedMaterialEvaluator
from pprint import pprint
from azure.storage.blob import BlobServiceClient
import uuid, json

import argparse
import json
import yaml

file_suffix = str(uuid.uuid4().hex)
print("Run GUID: ")
print(file_suffix)
all_pairs_path = f"query_response_pairs_{file_suffix}.jsonl"
all_evals = f"eval_results_{file_suffix}.txt"
'''
Code for incorporating key vault secrets.
'''
import os
from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# e.g. "https://my-keyvault.vault.azure.net/"
VAULT_URL = "https://eval-agent-kv.vault.azure.net/"

_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
_secret_client = SecretClient(vault_url=VAULT_URL, credential=_credential)

def get_secret(name: str, *, version: Optional[str] = None) -> str:
    """Return a secret value from Azure Key Vault."""
    return _secret_client.get_secret(name, version=version).value

def load_env_from_keyvault(mapping: dict[str, str]) -> None:
    """
    Given a mapping of ENV_VAR -> KEYVAULT_SECRET_NAME,
    fetch each secret and set os.environ[ENV_VAR].
    """
    for env_var, secret_name in mapping.items():
        value = get_secret(secret_name)
        os.environ[env_var] = value
load_env_from_keyvault({
    "AZURE_OPENAI_ENDPOINT":        "az-openai-endpoint",
    "AZURE_OPENAI_API_KEY":         "az-openai-api-key",
    "AZURE_OPENAI_API_VERSION":     "az-openai-api-version",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
    # Optional: storage secrets if you use keys/conn-strings
    # "AZURE_STORAGE_CONNECTION_STRING": "storage-connstring",
})

def load_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"⚠️ Config file '{path}' not found. Using defaults.")
        return {}
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f) or {}

def merge_config(defaults: dict, overrides: dict) -> dict:
    """Recursively merge user config over defaults."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in defaults:
            defaults[k] = merge_config(defaults[k], v)
        else:
            defaults[k] = v
    return defaults

# CLI argument for config
parser = argparse.ArgumentParser(description="Run E2E Agent Evaluation with optional config")
parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
args = parser.parse_args()

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")


print(openai_endpoint)
print(model_name)
print(deployment)
print(api_version)
print(openai_key)

# --- Load defaults from environment / hardcoded ---
defaults = {
    "project": {
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
    "project_name": "shayakproject",
    "resource_group_name": "shayak-test-rg",
},
    "simulators": ["adversarial", "direct", "indirect"],
    "evals": [
        "tool_call_accuracy", "intent_resolution", "task_adherence",
        "violence", "relevance", "coherence", "fluency", "self_harm",
        "sexual", "hate_unfairness", "code_vulnerability", "indirect_attack", "protected_material"
    ],
    "key_vault_uri": "",
    "custom_prompts": []
}

# --- Merge user config ---
user_config = load_config(args.config)
config = merge_config(defaults, user_config)

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

subscription_id = config["project"]["subscription_id"]
project_name = config["project"]["project_name"]
resource_group_name = config["project"]["resource_group_name"]

azure_ai_project = {
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
            "project_name": "shayakproject",
            "resource_group_name": "shayak-test-rg",
        }


print("subscription_id", config["project"]["subscription_id"])
print("project_name", config["project"]["project_name"])
print("resource_group_name", config["project"]["resource_group_name"])

print(azure_ai_project)

credential=DefaultAzureCredential()
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=openai_endpoint,
    api_key=openai_key
)

model_config = {
    "azure_endpoint": openai_endpoint,
    "azure_deployment": deployment,
}

def example_application_response(query: str, context: str) -> str:

    # Get a client handle for the AOAI model
    client = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_version=api_version,
        api_key=openai_key
    )

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": f"You are a user assistant who helps answer questions based on some context.\n\nContext: '{context}'",
        },
        {"role": "user", "content": query},
    ]
    # Call the model
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=10000
    )
    message = completion.to_dict()["choices"][0]["message"]
    if isinstance(message, dict):
        message = message["content"]
    return message

'''
Calls the agent, formats an Open-AI style message and appends it back to the conversation,
hence running the attacks end to end. 
'''
async def custom_simulator_callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    application_input = latest_message["content"]
    context = latest_message.get("context", None)
    # call your endpoint or ai application here
    response = example_application_response(query=application_input, context=context)
    # we are formatting the response to follow the openAI chat protocol format
    message = {
        "content": response,
        "role": "assistant",
        "context": context,
    }
    messages["messages"].append(message)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

def upload_to_blob(container_name: str, file_paths: list[str]) -> None:
    """
    Uploads the given files to an Azure Blob container.
    Storage account name is hardcoded.
    """
    # --- Hardcoded storage account name ---
    STORAGE_ACCOUNT_NAME = "evalsofagents"  # change this
    account_url = f"https://evalsofagents.blob.core.windows.net"

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
    except Exception:
        pass

    for file_path in file_paths:
        blob_name = os.path.basename(file_path)
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"✅ Uploaded: {blob_name}")
        
global list_of_prompts;
list_of_prompts = []

credential = DefaultAzureCredential()
token = credential.get_token("https://management.azure.com/.default")
print("Token tenant:", token.token.split(".")[1])  # base64 decode if needed

custom_simulator =  AdversarialSimulator(credential=DefaultAzureCredential(), azure_ai_project=azure_ai_project)
direct_attack_simulator = DirectAttackSimulator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential())
indirect_attack_simulator=IndirectAttackSimulator(azure_ai_project=azure_ai_project, credential=credential)

import asyncio

async def get_output_prompts_adv(scenario):
    try:
        outputs = await custom_simulator(
            scenario=scenario, max_simulation_results=10,
            target=custom_simulator_callback
        )
    
        # assuming outputs.to_eval_qr_json_lines() returns a str with multiple JSON objects separated by newlines
        json_lines = outputs.to_eval_qr_json_lines().splitlines()
    
        for line in json_lines:
            obj = json.loads(line)
            # adjust the key if it's "query" instead of "prompt"
            list_of_prompts.append(obj.get("query"))
    except Exception as expp:
        print("Exception occured")
        print(expp)

async def get_output_prompts_adv(scenario):
    try:
        outputs = await custom_simulator(
            scenario=scenario, max_simulation_results=10,
            target=custom_simulator_callback
        )
    
        # assuming outputs.to_eval_qr_json_lines() returns a str with multiple JSON objects separated by newlines
        json_lines = outputs.to_eval_qr_json_lines().splitlines()
    
        for line in json_lines:
            obj = json.loads(line)
            # adjust the key if it's "query" instead of "prompt"
            list_of_prompts.append(obj.get("query"))
    except Exception as expp:
        print("Exception occurred");
        print(expp)
    
async def get_output_prompts_da():
    try:
        outputs = await direct_attack_simulator(
            scenario=AdversarialScenario.ADVERSARIAL_CONVERSATION, max_simulation_results=10,
            target=custom_simulator_callback
        )
    
        # assuming outputs.to_eval_qr_json_lines() returns a str with multiple JSON objects separated by newlines
        json_lines = outputs.to_eval_qr_json_lines().splitlines()
    
        for line in json_lines:
            obj = json.loads(line)
            
            # adjust the key if it's "query" instead of "prompt"
            list_of_prompts.append(obj.get("query"))
        try:
            print("Direct Jail Break Attacks")
        except Exception as e:
            print(e)
    except Exception as expp:
        print("Exception in da prompt")
        print(expp)
    
async def get_output_prompts_ida():
    try:
        outputs = await indirect_attack_simulator(
            max_simulation_results=10,
            target=custom_simulator_callback,
            max_conversation_turns = 3
        )
    
        # assuming outputs.to_eval_qr_json_lines() returns a str with multiple JSON objects separated by newlines
        json_lines = outputs.to_eval_qr_json_lines().splitlines()
    
        for line in json_lines:
            obj = json.loads(line)
            # adjust the key if it's "query" instead of "prompt"
            list_of_prompts.append(obj.get("query"))
    except Exception as expp:
        print("Exception!")
        print(expp)
    
    print(list_of_prompts)
    
'''
async def main():
    simulators_to_run = [s.lower() for s in config["simulators"]]
    if "adversarial" in simulators_to_run:
        scenarios = [
            AdversarialScenario.ADVERSARIAL_QA,
            AdversarialScenario.ADVERSARIAL_CONVERSATION,
            AdversarialScenario.ADVERSARIAL_SUMMARIZATION,
            AdversarialScenario.ADVERSARIAL_SEARCH,
            AdversarialScenario.ADVERSARIAL_REWRITE,
            AdversarialScenario.ADVERSARIAL_CONTENT_GEN_UNGROUNDED,
            AdversarialScenario.ADVERSARIAL_CONTENT_PROTECTED_MATERIAL
        ]

        for scenario in scenarios:
            await get_output_prompts_adv(scenario)  # each one isolated
    if "direct" in simulators_to_run:
        await get_output_prompts_da()
    if "indirect" in simulators_to_run:
        await get_output_prompts_ida()

    print("\n✅ All completed (some may have failed but others continued):")
    pprint(list_of_prompts, width=200)

asyncio.run(main())


load_dotenv()
'''

from azure.ai.projects import __version__ as projects_version
from packaging.version import Version
# some dependencies have been updated with breaking changes -- indicates whether to use updated models and APIs or not
updated_agents = Version(projects_version) > Version("1.0.0b10") or projects_version.startswith("1.0.0a")

credential=DefaultAzureCredential()

from azure.ai.agents.models import FunctionTool, ToolSet
project_client = AIProjectClient(
    endpoint="https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject",
    credential=DefaultAzureCredential()
)

agent = project_client.agents.get_agent(
    agent_id = "asst_40jxCEVxQniq4Pr7lDxTxeYu"
)

print(f"Fetched agent, ID: {agent.id}")
print("LIST OF PROMPTS")
print(list_of_prompts)
custom_prompts = config.get("custom_prompts", [])
print("List of custom prompts passed")
print(custom_prompts)
for prompt in custom_prompts:
    list_of_prompts.append(prompt)

print("FINAL LIST OF PROMPTS")
print(list_of_prompts)

tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
intent_resolution = IntentResolutionEvaluator(model_config=model_config)
task_adherence = TaskAdherenceEvaluator(model_config=model_config)
relevance = RelevanceEvaluator(model_config=model_config)
coherence = CoherenceEvaluator(model_config=model_config)
fluency = FluencyEvaluator(model_config=model_config)


violence = ViolenceEvaluator(credential=credential, azure_ai_project=azure_ai_project)
self_harm = SelfHarmEvaluator(credential=credential, azure_ai_project=azure_ai_project)
sexual = SexualEvaluator(credential=credential, azure_ai_project=azure_ai_project)
hate_unfairness= HateUnfairnessEvaluator(credential=credential, azure_ai_project=azure_ai_project)
code_vulnerability = CodeVulnerabilityEvaluator(credential=credential, azure_ai_project=azure_ai_project)
indirect_attack = IndirectAttackEvaluator(credential=credential, azure_ai_project=azure_ai_project)
protected_material = ProtectedMaterialEvaluator(credential=credential, azure_ai_project=azure_ai_project)


with open(f"list_of_prompts_{file_suffix}.txt", "w") as f:
    f.write(json.dumps(list_of_prompts))
try:
    print("Uploading list of prompts information to azure storage")
    upload_to_blob(container_name="list-of-prompts-1", file_paths=[f"list_of_prompts_{file_suffix}.txt"])
except Exception as expp:
    print("Exception in uploading list of prompts")
    print(expp)
    
count = 0 
for prompt in list_of_prompts:
    try:
        count += 1; 
        thread = project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")
        
        # Create message to thread
        
        MESSAGE = prompt
        
        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=MESSAGE,
        )
            
        print(f"Created message, ID: {message.id}")
        
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
        
        print(f"Run finished with status: {run.status}")
        
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")
        
        print(f"Run ID: {run.id}")
    
        '''
        print("==============================================================LIST OF MESSAGES FOR THE CONVERSATION ===========================================================================")
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
        '''
        
        import json
        from azure.ai.evaluation import AIAgentConverter
        
        # Initialize the converter that will be backed by the project.
        converter = AIAgentConverter(project_client)
        
        thread_id = thread.id
        run_id = run.id
    
        print("==============================================================CONVERTED DATA===========================================================================")
        converted_data = converter.convert(thread_id=thread_id, run_id=run_id)
        print(converted_data)
        jsonl_handle = open(all_pairs_path, "a", encoding="utf-8")

        try:
            # If the converter returns a list of records, write each as its own line.
            if isinstance(converted_data, list):
                for rec in converted_data:
                    jsonl_handle.write(json.dumps(rec, default=str) + "\n")
            else:
                # Otherwise assume it's a single dict-like record.
                jsonl_handle.write(json.dumps(converted_data, default=str) + "\n")
        except Exception as write_err:
            print("Failed to append converted_data to JSONL:", write_err)

        # Save the converted data to a JSONL file
    
        file_name = "evaluationDataAdverserialData" + str(count) + ".jsonl"
        evaluation_data = converter.prepare_evaluation_data(thread_ids=thread.id, filename=file_name)
        
        load_dotenv()
        
        
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=openai_endpoint,
            api_key=openai_key,
            api_version=api_version,
            azure_deployment=deployment,
        )
        # Needed to use content safety evaluators
        azure_ai_project = {
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
            "project_name": "shayakproject",
            "resource_group_name": "shayak-test-rg",
        }

        evaluator_map = {
            "tool_call_accuracy": tool_call_accuracy,
            "intent_resolution": intent_resolution,
            "task_adherence": task_adherence,
            "violence": violence,
            "relevance": relevance,
            "coherence": coherence,
            "fluency": fluency,
            "self_harm": self_harm,
            "sexual": sexual,
            "hate_unfairness": hate_unfairness,
            "code_vulnerability": code_vulnerability,
            "indirect_attack": indirect_attack,
            "protected_material": protected_material,
        }

        for name, fn in evaluator_map.items():
            if name in config["evals"]:
                fn(query=converted_data['query'], response=converted_data['response'])

        active_evaluators = {k: v for k, v in evaluator_map.items() if k in config["evals"]}

        response = evaluate(
            data=file_name,
            evaluators=active_evaluators,
            azure_ai_project="https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject"
        )
        
        pprint(list_of_prompts, width=200)
        pprint(f'Azure ML Studio URL: {response.get("studio_url")}')
        pprint(response)

        evals_handle = open(all_evals, "a", encoding="utf-8")

        try:
            evals_handle.write(str(response) + "\n")
        except Exception as write_err:
            print("Failed to append evals to txt file:", write_err)

    except Exception as exception:
        print("exception occured!")
        print(exception)
        continue;


try:
    print("Uploading the query-response pairs to the storage account")
    upload_to_blob(container_name="query-response-pairs-1", file_paths=[f"query_response_pairs_{file_suffix}.jsonl"])
    print("Uploaded query json pairs results!")
except Exception as expp: 
    print(expp)
    
try:
    print("Uploading all evals for this run to azure storage")
    upload_to_blob(container_name="list-of-evals-1", file_paths=[all_evals])
except Exception as expp:
    print("Exception in uploading all evals")
    print(expp)
