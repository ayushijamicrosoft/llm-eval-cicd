#!/usr/bin/env python3
"""
Single-file, refactored version of your red-team script.
Same behavior as before; reorganized into functions for clarity.
Run: python redteam_singlefile.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import pprint
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd

# Azure + OpenAI imports (keep same usage)
from azure.ai.evaluation import (
    evaluate,
    GroundednessEvaluator,
    ToolCallAccuracyEvaluator,
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    TaskAdherenceEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    ViolenceEvaluator,
    SelfHarmEvaluator,
    SexualEvaluator,
    HateUnfairnessEvaluator,
    CodeVulnerabilityEvaluator,
    IndirectAttackEvaluator,
    ProtectedMaterialEvaluator,
)
from azure.ai.evaluation.simulator import (
    AdversarialSimulator,
    AdversarialScenario,
    DirectAttackSimulator,
    IndirectAttackSimulator,
    Simulator,
)
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
from azure.ai.projects import AIProjectClient

from azure.identity import (
    DefaultAzureCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from openai import AzureOpenAI
from dotenv import load_dotenv

# keep duplicate-safe imports (your original file had repeated imports)
import importlib.resources as pkg_resources
import uuid as _uuid

# -------------------------
# Globals kept from original
# -------------------------
file_suffix = _uuid.uuid4().hex
print("Run GUID:", file_suffix)

# Default VAULT URL (kept same)
VAULT_URL = "https://eval-agent-kv.vault.azure.net/"

# store collected prompts similar to original
list_of_prompts: List["PromptRecord"] = []

# environment placeholders (will be overridden by keyvault loader or env)
openai_endpoint: Optional[str] = None
model_name: Optional[str] = None
deployment: Optional[str] = None
openai_key: Optional[str] = None
api_version: Optional[str] = None

# -------------------------
# Data classes / small utils
# -------------------------
@dataclass
class PromptRecord:
    prompt: str
    simulator: str
    scenario: Optional[str] = None


# -------------------------
# Config helpers (kept behavior)
# -------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"⚠️ Config file '{path}' not found. Using defaults.")
        return {}
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def merge_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge user config over defaults (same behavior)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in defaults and isinstance(defaults[k], dict):
            defaults[k] = merge_config(defaults[k], v)
        else:
            defaults[k] = v
    return defaults


# -------------------------
# KeyVault + env loader (kept behavior)
# -------------------------
_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
_secret_client = SecretClient(vault_url=VAULT_URL, credential=_credential)


def get_secret(name: str, *, version: Optional[str] = None) -> str:
    """Return a secret value from Azure Key Vault."""
    return _secret_client.get_secret(name, version=version).value


def load_env_from_keyvault(mapping: Dict[str, str]) -> None:
    """
    Given a mapping of ENV_VAR -> KEYVAULT_SECRET_NAME,
    fetch each secret and set os.environ[ENV_VAR].
    """
    for env_var, secret_name in mapping.items():
        try:
            value = get_secret(secret_name)
            os.environ[env_var] = value
        except Exception as e:
            # preserve original behavior (no raise) but log error
            print(f"Could not load secret '{secret_name}' for env '{env_var}': {e!s}")


# -------------------------
# Blob upload helper (kept behavior)
# -------------------------
def upload_to_blob(container_name: str, file_paths: List[str]) -> None:
    STORAGE_ACCOUNT_NAME = "evalsofagents"
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        # ignore container exists / permission issues similar to original except pass
        pass

    for file_path in file_paths if isinstance(file_paths, list) else [file_paths]:
        try:
            blob_name = os.path.basename(file_path)
            with open(file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            print(f"Uploaded: {blob_name}")
        except Exception as e:
            print(f"Failed uploading {file_path} to container {container_name}: {e!s}")


# -------------------------
# OpenAI / Azure OpenAI helpers
# -------------------------
def create_openai_client(api_version: str | None = None, azure_endpoint: str | None = None, api_key: str | None = None) -> AzureOpenAI:
    """Create AzureOpenAI client (keeps old signature and behavior)."""
    return AzureOpenAI(
        api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
    )


def example_application_response(query: str, context: str, deployment_name: Optional[str] = None) -> str:
    client = create_openai_client(api_version=api_version, azure_endpoint=openai_endpoint, api_key=openai_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a user assistant who helps answer questions based on some context.\n\n"
                f"Context: '{context}'"
            ),
        },
        {"role": "user", "content": query},
    ]
    completion = client.chat.completions.create(
        model=deployment_name or deployment,
        messages=messages,
        max_completion_tokens=10000,
    )
    message = completion.to_dict()["choices"][0]["message"]
    if isinstance(message, dict):
        message = message.content
    return message

SIMULATOR_EVALUATOR_MAP = {
    "hate_unfairness": [
        "hate_unfairness"
    ],
    "sexual": ["sexual"
    ],
    "self_harm": [
        "self_harm",
    ],
    "violence": [
        "violence"
    ],
}

# -------------------------
# Simulator / callback (kept behavior)
# -------------------------
async def custom_simulator_callback(
    messages: Dict[str, Any],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print("Messages", messages)
    # messages is expected to have list-like structure; keep original access
    latest_message = messages[-1]
    application_input = latest_message.content if hasattr(latest_message, "content") else latest_message.get("content", "")
    context_from_message = "Talk to the user."
    response = example_application_response(query=application_input, context=context_from_message)
    message = {
        "content": response,
        "role": "assistant",
        "context": context_from_message,
    }
    # maintain the same structure additions as original
    try:
        messages["messages"].append(message)
        return {
            "messages": messages["messages"],
            "stream": stream,
            "session_state": session_state,
            "context": context_from_message,
        }
    except Exception:
        # Some simulators pass plain lists - handle that too
        if isinstance(messages, list):
            messages.append(message)
            return {"messages": messages, "stream": stream, "session_state": session_state, "context": context_from_message}
        raise


# -------------------------
# Evaluator builder (kept behavior)
# -------------------------
def build_evaluators(
    model_config: AzureOpenAIModelConfiguration,
    credential: DefaultAzureCredential | None = None,
    AZURE_AI_PROJECT: Optional[str] = None,
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
    # keep UngroundedAttributesEvaluator as was referenced previously if available; otherwise comment-safe
    try:
        from azure.ai.evaluation import UngroundedAttributesEvaluator

        ungrounded_attributes = UngroundedAttributesEvaluator(**content_safety_kwargs)
    except Exception:
        ungrounded_attributes = None

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

def process_prompts_with_agent(
    prompt_records: List[PromptRecord],
    project_client: AIProjectClient,
    project_client_evals: AIProjectClient,
    agent_id: str,
    evaluator_map: Dict[str, Any],
    enabled_evals: List[str],
    all_pairs_path: str,
    evaluation_data_file: str,
    run_guid: str
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

# -------------------------
# Azure OpenAI callback wrapper (kept behavior)
# -------------------------
async def azure_openai_callback(
    messages: list,
    stream: Optional[bool] = False,  # noqa: ARG001
    session_state: Optional[str] = None,  # noqa: ARG001
    context: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> dict[str, list[dict[str, str]]]:
    # Token provider for Azure AD auth
    token_provider = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")

    client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_version=azure_openai_api_version,
        azure_ad_token_provider=token_provider,
    )

    # Normalize messages to simpler dicts
    messages_list = [{"role": getattr(message, "role", message.get("role")), "content": getattr(message, "content", message.get("content"))} for message in messages]
    latest_message = messages_list[-1]["content"]

    try:
        response = client.chat.completions.create(
            model=azure_openai_deployment,
            messages=[{"role": "user", "content": latest_message}],
            max_completion_tokens=500,
        )
        formatted_response = {"content": response.choices[0].message.content, "role": "assistant"}
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e!s}")
        formatted_response = {"content": "I encountered an error and couldn't process your request.", "role": "assistant"}
    return {"messages": [formatted_response]}


# -------------------------
# Utility to handle sync/async API behavior (kept behavior)
# -------------------------
async def maybe_await(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


# -------------------------
# Main orchestration (kept behavior)
# -------------------------
async def main_async(config_path: Optional[str] = None):
    global openai_endpoint, model_name, deployment, openai_key, api_version
    global azure_openai_deployment, azure_openai_endpoint, azure_openai_api_key, azure_openai_api_version
    global AZURE_AI_PROJECT

    # CLI args were parsed earlier; but keep function param for programmatic use
    # --- Load defaults from environment / hardcoded ---
    defaults = {
        "project": {
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
            "project_name": "shayakproject",
            "resource_group_name": "shayak-test-rg",
        },
        "simulators": ["adversarial", "direct", "indirect"],
        "evals": [
            "tool_call_accuracy",
            "intent_resolution",
            "task_adherence",
            "violence",
            "relevance",
            "coherence",
            "fluency",
            "self_harm",
            "sexual",
            "hate_unfairness",
            "code_vulnerability",
            "indirect_attack",
            "protected_material",
        ],
        "key_vault_uri": "",
        "custom_prompts": [],
    }

    # load user config
    user_config = load_config(config_path)
    config = merge_config(defaults, user_config)

    # If the user used KeyVault loader earlier, attempt to populate env vars exactly as original
    load_env_from_keyvault(
        {
            "AZURE_OPENAI_ENDPOINT": "az-openai-endpoint",
            "AZURE_OPENAI_API_KEY": "az-openai-api-key",
            "AZURE_OPENAI_API_VERSION": "az-openai-api-version",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
        }
    )

    # mirror original env assignment behaviour
    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

    # set some environment vars consistent with original
    os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint or ""
    os.environ["AZURE_DEPLOYMENT_NAME"] = deployment or ""
    os.environ["AZURE_API_VERSION"] = api_version or ""
    os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

    subscription_id = config["project"]["subscription_id"]
    project_name = config["project"]["project_name"]
    resource_group_name = config["project"]["resource_group_name"]

    # Additional Azure/OpenAI variables used later in original
    azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", deployment)
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", openai_endpoint)
    azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", openai_key)
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", api_version)

    # create credentials similar to original
    credential = AzureCliCredential()

    AZURE_AI_PROJECT = "https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject"

    # Create the `RedTeam` instance with minimal configurations (same behavior)
    red_team = RedTeam(
        azure_ai_project=AZURE_AI_PROJECT,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness],
        num_objectives=1,
    )

    # Run example callback scan (keeps naming and behavior)
    print("Starting BASIC CALLBACK SCAN")
    try:
        result = await red_team.scan(
            target=custom_simulator_callback,
            scan_name="Basic-Callback-Scan",
            attack_strategies=[AttackStrategy.Flip],
            output_path="red_team_output.json",
        )
    except TypeError:
        # some SDKs expose sync scan - support both
        result = red_team.scan(
            target=custom_simulator_callback,
            scan_name="Basic-Callback-Scan",
            attack_strategies=[AttackStrategy.Flip],
            output_path="red_team_output.json",
        )

    print("BASIC RESULT -------------------------------------------------------------------------------")
    try:
        print(result.attack_details)
        attack_details_list = result.attack_details
        for attack_detail in attack_details_list:
            try:
                list_of_prompts.append(
                    PromptRecord(
                        prompt=attack_detail["conversation"][0]["content"],
                        scenario=attack_detail.get("attack_technique"),
                        simulator=attack_detail.get("risk_category"),
                    ))
            except: 
                print("continue")
    except Exception:
        print("No attack_details available or unexpected result format")
    print("Basic scan done:", result)
    print(list_of_prompts)

    # advanced model red team scan (keeps behavior and uses file_suffix)
    model_red_team = RedTeam(
        azure_ai_project=AZURE_AI_PROJECT,
        credential=credential,
        risk_categories=[RiskCategory.Violence, RiskCategory.HateUnfairness, RiskCategory.Sexual, RiskCategory.SelfHarm],
        num_objectives=5,
    )

    try:
        advanced_result = await maybe_await(
            model_red_team.scan,
            azure_openai_callback,
            scan_name="Advanced-Callback-Scan",
            attack_strategies=[AttackStrategy.EASY],
            output_path=f"Advanced-Callback-Scan_{file_suffix}.json",
        )
    except Exception as e:
        print(f"Advanced scan failed: {e!s}")
        advanced_result = None
    
    # attempt upload similar to original
    try:
        upload_to_blob("advanced-red-teaming-results", [f"Advanced-Callback-Scan_{file_suffix}.json"])
    except Exception as e:
        print(f"Upload attempt failed: {e!s}")

    print("ADVANCED RESULT -------------------------------------------------------------------------------")
    
    if advanced_result is not None:
        try:
            print(advanced_result.attack_details)
            attack_details_list = advanced_result.attack_details
            for attack_detail in attack_details_list:
                try:
                    list_of_prompts.append(
                        PromptRecord(
                            prompt=attack_detail["conversation"][0]["content"],
                            scenario=attack_detail.get("attack_technique"),
                            simulator=attack_detail.get("risk_category"),
                        ))
                except: 
                    print("continue")
        except Exception:
            print("No attack_details available or unexpected result format")
            print("Advanced scan done:", advanced_result)
        except Exception:
            pprint.pprint(advanced_result)
    else:
        print("No advanced result available")

    # final prints (keep original debug prints)
    print(openai_endpoint)
    print(model_name)
    print(deployment)
    print(api_version)
    print(openai_key)
    print("FINAL LIST OF PROMPTS")
    print(list_of_prompts)


def main():
    parser = argparse.ArgumentParser(description="Run AI Red Teaming with optional config")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
    args = parser.parse_args()

    # call main async and block until completion
    asyncio.run(main_async(config_path=args.config))


if __name__ == "__main__":
    main()
    project_client = AIProjectClient(
        endpoint="https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject",
        credential=DefaultAzureCredential()
    )
    
    agent = project_client.agents.get_agent(
        agent_id = "asst_40jxCEVxQniq4Pr7lDxTxeYu"
    )
    credential = DefaultAzureCredential()
    eval_model_config = AzureOpenAIModelConfiguration(
        azure_endpoint="https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25",
        api_key="EAX2Sbse7EkMn12uz8FQEMJO2TrSICmivBlgWkQWZc6utw2HWneXJQQJ99BIACHYHv6XJ3w3AAAAACOG7l1l",
        api_version=api_version,
        azure_deployment=deployment,
    )
    evaluator_map: Dict[str, Any] = build_evaluators(model_config=eval_model_config, credential=credential)
    
    project_client = AIProjectClient(
        endpoint="https://shayak-foundry.services.ai.azure.com/api/projects/shayakproject",
        credential=DefaultAzureCredential())
    all_pairs_path = f"query_response_pairs_{file_suffix}.jsonl"
    evaluation_data_file = "evaluationDataRedTeaming.jsonl"
    
    process_prompts_with_agent(
        prompt_records=list_of_prompts,
        project_client=project_client,
        agent_id=agent.id,
        evaluator_map=evaluator_map,
        enabled_evals=[],
        all_pairs_path=all_pairs_path,
        evaluation_data_file=evaluation_data_file,
        run_guid=file_suffix,
        )
