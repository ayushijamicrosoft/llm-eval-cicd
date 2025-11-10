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
parser = argparse.ArgumentParser(description="Run AI Red Teaming with optional config")
parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
args = parser.parse_args()

openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

def build_evaluators(
    model_config: AzureOpenAIModelConfiguration,
    credential: DefaultAzureCredential,
) -> Dict[str, Any]:
    if AZURE_AI_PROJECT is None:
        raise RuntimeError("AZURE_AI_PROJECT is not initialized")

    content_safety_kwargs = {"credential": credential, "azure_ai_project": AZURE_AI_PROJECT}

    tool_call_accuracy = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_resolution = IntentResolutionEvaluator(model_config=model_config)
    task_adherence = TaskAdherenceEvaluator(model_config=model_config)
    relevance = RelevanceEvaluator(model_config=model_config)
    coherence = CoherenceEvaluator(model_config=model_config)
    fluency = FluencyEvaluator(model_config=model_config)

    violence = ViolenceEvaluator(**content_safety_kwargs)
    self_harm = SelfHarmEvaluator(**content_safety_kwargs)
    sexual = SexualEvaluator(**content_safety_kwargs)
    hate_unfairness = HateUnfairnessEvaluator(**content_safety_kwargs)
    code_vulnerability = CodeVulnerabilityEvaluator(**content_safety_kwargs)
    indirect_attack = IndirectAttackEvaluator(**content_safety_kwargs)
    protected_material = ProtectedMaterialEvaluator(**content_safety_kwargs)
    ungrounded_attributes = UngroundedAttributesEvaluator(**content_safety_kwargs)

    evaluator_map: Dict[str, Any] = {
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
        "ungrounded_attributes": ungrounded_attributes,
    }
    return evaluator_map


def run_selected_evaluators(
    evaluator_map: Dict[str, Any],
    eval_names: List[str],
    converted_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run evaluators and return a mapping of evaluator_name -> result.
    The result is whatever the evaluator callable returns (often a dict).
    """
    results: Dict[str, Any] = {}
    for name in eval_names:
        fn = evaluator_map.get(name)
        if not fn:
            continue

        try:
            if name == "ungrounded_attributes":
                # This evaluator expects either `conversation` OR individual inputs.
                # We try to pass all three if possible.
                res = fn(
                    query=converted_data.get("query", ""),
                    context=open("Workload_Register_prompts.txt").read(),
                    response=converted_data.get("response", ""),
                )
            else:
                res = fn(
                    query=converted_data.get("query", ""),
                    response=converted_data.get("response", ""),
                )

            results[name] = res
        except Exception as exc:
            print(f"Evaluator {name} failed for query: {exc}")
            results[name] = {"error": str(exc)}
    return results

# ---------------------------------------------------------------------------
# Agent run and conversion
# ---------------------------------------------------------------------------

def run_agent_for_prompt(
    project_client: AIProjectClient,
    agent_id: str,
    prompt: str,
) -> tuple[str, str]:
    thread = project_client.agents.threads.create()
    print(f"Created thread, ID: {thread.id}")

    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    print(f"Created message, ID: {message.id}")

    run = project_client.agents.runs.create(thread_id=thread.id, agent_id=agent_id)

    while run.status in ["queued", "in_progress", "requires_action"]:
        time.sleep(1)
        run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)

        if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            if not tool_calls:
                print("No tool calls provided - cancelling run")
                project_client.agents.runs.cancel(thread_id=thread.id, run_id=run.id)
                break

            tool_outputs: List[ToolOutput] = []
            for tool_call in tool_calls:
                if isinstance(tool_call, RequiredFunctionToolCall):
                    try:
                        print(f"Executing tool call: {tool_call}")
                        # define `functions.execute` in your codebase
                        output = functions.execute(tool_call)  # type: ignore[name-defined]
                        tool_outputs.append(
                            ToolOutput(
                                tool_call_id=tool_call.id,
                                output=output,
                            )
                        )
                    except Exception as e:
                        print(f"Error executing tool_call {tool_call.id}: {e}")

            if tool_outputs:
                project_client.agents.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )

    print(f"Run finished with status: {run.status}")
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    return thread.id, run.id


def append_converted_data_to_jsonl(all_pairs_path: str, converted_data: Any) -> None:
    with open(all_pairs_path, "a", encoding="utf-8") as jsonl_handle:
        try:
            if isinstance(converted_data, list):
                for rec in converted_data:
                    jsonl_handle.write(json.dumps(rec, default=str) + "\n")
            else:
                jsonl_handle.write(json.dumps(converted_data, default=str) + "\n")
        except Exception as write_err:
            print("Failed to append converted_data to JSONL:", write_err)


def process_prompts_with_agent(
    prompt_records: List[PromptRecord],
    project_client: AIProjectClient,
    agent_id: str,
    evaluator_map: Dict[str, Any],
    enabled_evals: List[str],
    all_pairs_path: str,
    evaluation_data_file: str,
    run_guid: str,
) -> None:
    converter = AIAgentConverter(project_client)
    pair_files: set[str] = set()

    for record in prompt_records:
        try:
            print("=" * 80)
            print(f"Simulator: {record.simulator}, scenario: {record.scenario}")
            print(f"Prompt: {record.prompt}")

            thread_id, run_id = run_agent_for_prompt(project_client, agent_id, record.prompt)

            converted_data = converter.convert(thread_id=thread_id, run_id=run_id)
            print("Converted data:")
            print(converted_data)

            append_converted_data_to_jsonl(all_pairs_path, converted_data)

            converter.prepare_evaluation_data(thread_ids=thread_id, filename=evaluation_data_file)

            if isinstance(converted_data, dict):

                desired_eval_names = SIMULATOR_EVALUATOR_MAP.get(
                    record.simulator,
                    list(evaluator_map.keys()),
                )
            
                # Only require that the evaluator actually exists
                selected_eval_names = [
                    name
                    for name in desired_eval_names
                    if name in evaluator_map
                ]
    
                # run all evaluators once and capture results
                eval_results = run_selected_evaluators(evaluator_map, selected_eval_names, converted_data)

                # per simulator-evaluator pair file with shared run_guid,
                # now including evaluation_result
                for evaluator_name in selected_eval_names:
                    pair_file = f"sim_eval_pair_{record.simulator}_{evaluator_name}_{run_guid}.jsonl"
                    log_entry = {
                        "run_guid": run_guid,
                        "simulator": record.simulator,
                        "scenario": record.scenario,
                        "evaluator": evaluator_name,
                        "prompt": record.prompt,
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "evaluation_result": eval_results.get(evaluator_name),
                    }
                    with open(pair_file, "a", encoding="utf-8") as pf:
                        pf.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
                    pair_files.add(pair_file)

        except Exception as exc:
            print("Exception while processing prompt:")
            print(exc)
            continue

    # upload all simulator-evaluator pair files for this run
    if pair_files:
        try:
            print("Uploading sim-eval pair logs to azure storage")
            upload_to_blob(container_name="sim-eval-pairs-1", file_paths=list(pair_files))
        except Exception as exc:
            print("Exception uploading sim-eval pairing logs")
            print(exc)



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

from typing import Optional, Dict, Any
import os

# Azure imports
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy

# OpenAI imports
from openai import AzureOpenAI

# Azure Credential imports
from azure.identity import AzureCliCredential, get_bearer_token_provider

# Initialize Azure credentials
credential = AzureCliCredential()

azure_ai_project = "https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject"

# Azure OpenAI deployment information
azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")  # e.g., "gpt-4"
azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")  # e.g., "your-api-key"
azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")  # Use the latest API version

# Define a simple callback function that always returns a fixed response
def financial_advisor_callback(query: str) -> str:  # noqa: ARG001
    return "I'm a financial advisor assistant. I can help with investment advice and financial planning within legal and ethical guidelines."

# Create the `RedTeam` instance with minimal configurations
red_team = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=credential,
    risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness],
    num_objectives=1,
)

import asyncio
import inspect
from pathlib import Path
# ... keep all your other imports and helpers above ...

# Define a callback that uses Azure OpenAI API to generate responses
async def azure_openai_callback(
    messages: list,
    stream: Optional[bool] = False,  # noqa: ARG001
    session_state: Optional[str] = None,  # noqa: ARG001
    context: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> dict[str, list[dict[str, str]]]:
    # Get token provider for Azure AD authentication
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_version=azure_openai_api_version,
        azure_ad_token_provider=token_provider,
    )

    ## Extract the latest message from the conversation history
    messages_list = [{"role": message.role, "content": message.content} for message in messages]
    latest_message = messages_list[-1]["content"]

    try:
        # Call the model
        response = client.chat.completions.create(
            model=azure_openai_deployment,
            messages=[
                {"role": "user", "content": latest_message},
            ],
            # max_tokens=500, # If using an o1 base model, comment this line out
            max_completion_tokens=500,  # If using an o1 base model, uncomment this line
            # temperature=0.7, # If using an o1 base model, comment this line out (temperature param not supported for o1 base models)
        )

        # Format the response to follow the expected chat protocol format
        formatted_response = {"content": response.choices[0].message.content, "role": "assistant"}
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e!s}")
        formatted_response = "I encountered an error and couldn't process your request."
    return {"messages": [formatted_response]}

async def main():
    # initialize credential, clients, callbacks etc. here (same as your top-level code)
    credential = AzureCliCredential()
    azure_ai_project = "https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25"
    red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness],
        num_objectives=1,
    )

    # Example target (callback) run
    result = await red_team.scan(
        target=financial_advisor_callback,
        scan_name="Basic-Callback-Scan",
        attack_strategies=[AttackStrategy.Flip],
        output_path="red_team_output.json",
    )
    print("Basic scan done:", result)

    # If you want to call a scan that might be synchronous on some versions of the SDK,
    # you can defensively handle both coroutine and normal function:
    # (optional helper)
    async def maybe_await(fn, *args, **kwargs):
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

    # Example: run another scan using the helper (handles sync or async scan implementations)
    model_red_team = RedTeam(
        azure_ai_project=azure_ai_project,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=5,
    )
    advanced_result = await maybe_await(
        model_red_team.scan,
        target=azure_openai_callback,
        scan_name="Advanced-Callback-Scan",
        attack_strategies=[AttackStrategy.EASY],  # keep minimal for demo
        output_path="Advanced-Callback-Scan.json",
    )
    print("Advanced scan done:", advanced_result)


asyncio.run(main())

async def main2():
    # Define a model configuration to test
    azure_oai_model_config = {
        "azure_endpoint": azure_openai_endpoint,
        "azure_deployment": azure_openai_deployment,
        "api_key": azure_openai_api_key,
    }
    
    # Run the red team scan called "Intermediary-Model-Target-Scan"
    result = await red_team.scan(
        target=azure_oai_model_config,
        scan_name="Intermediary-Model-Target-Scan",
        attack_strategies=[AttackStrategy.Flip],
    )
    risk_category = [RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm]
    for r_cat in risk_category:
        # Create the RedTeam instance with all of the risk categories with 5 attack objectives generated for each category
        model_red_team = RedTeam(
            azure_ai_project=azure_ai_project,
            credential=credential,
            risk_categories=[r_cat],
            num_objectives=20,
        )
        
        # Run the red team scan with multiple attack strategies
        advanced_result = await model_red_team.scan(
            target=azure_openai_callback,
            scan_name="Advanced-Callback-Scan",
            attack_strategies=[
                AttackStrategy.EASY,  # Group of easy complexity attacks
                AttackStrategy.MODERATE,  # Group of moderate complexity attacks
                AttackStrategy.CharacterSpace,  # Add character spaces
                AttackStrategy.ROT13,  # Use ROT13 encoding
                AttackStrategy.UnicodeConfusable,  # Use confusable Unicode characters
                AttackStrategy.CharSwap,  # Swap characters in prompts
                AttackStrategy.Morse,  # Encode prompts in Morse code
                AttackStrategy.Leetspeak,  # Use Leetspeak
                AttackStrategy.Url,  # Use URLs in prompts
                AttackStrategy.Binary,  # Encode prompts in binary
                AttackStrategy.Compose([AttackStrategy.Base64, AttackStrategy.ROT13]),  # Use two strategies in one attack
            ],
            output_path=f"Advanced-Callback-Scan_{r_cat}.json",
        )
        
        path_to_prompts = ".\data\prompts.json"
        
        # Create the RedTeam specifying the custom attack seed prompts to use as objectives
        custom_red_team = RedTeam(
            azure_ai_project=azure_ai_project,
            credential=credential,
            custom_attack_seed_prompts=path_to_prompts,  # Path to a file containing custom attack seed prompts
        )
        
        custom_red_team_result = await custom_red_team.scan(
            target=azure_openai_callback,
            scan_name="Custom-Prompt-Scan",
            attack_strategies=[
                AttackStrategy.EASY,  # Group of easy complexity attacks
                AttackStrategy.MODERATE,  # Group of moderate complexity attacks
                AttackStrategy.DIFFICULT,  # Group of difficult complexity attacks
            ],
            output_path=f"Custom-Prompt-Scan_{r_cat}.json",
        )
        print(custom_red_team_results)

asyncio.run(main2())

