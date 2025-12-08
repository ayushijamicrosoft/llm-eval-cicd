import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

from openai import AzureOpenAI

from azure.ai.evaluation import (
    evaluate,
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
    AIAgentConverter,
    UngroundedAttributesEvaluator,
)
from azure.ai.evaluation.simulator import (
    AdversarialSimulator,
    AdversarialScenario,
    DirectAttackSimulator,
    IndirectAttackSimulator,
)
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FunctionTool,
    ListSortOrder,
    RequiredFunctionToolCall,
    SubmitToolOutputsAction,
    ToolOutput,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    prompt: str
    simulator: str
    scenario: Optional[str] = None


# Simulator -> evaluators mapping
SIMULATOR_EVALUATOR_MAP = {
    "adversarial": [
        "violence", "self_harm", "sexual",
        "hate_unfairness", "protected_material",
        "ungrounded_attributes", "relevance", "coherence", "fluency"
    ],
    "direct": [
        "violence", "self_harm", "sexual",
        "hate_unfairness", "protected_material",
        "tool_call_accuracy", "intent_resolution", "task_adherence",
        "ungrounded_attributes"
    ],
    "indirect": [
        "indirect_attack", "ungrounded_attributes",
        "violence", "self_harm", "sexual",
        "hate_unfairness", "protected_material",
        "task_adherence", "relevance"
    ],
    '''
    "custom": [
        "tool_call_accuracy", "intent_resolution", "task_adherence",
        "relevance", "coherence", "fluency",
        "code_vulnerability", "ungrounded_attributes",
    ],
    '''
    #hashed above to add custom prompts for direct attack
     "custom": [
        "tool_call_accuracy", "intent_resolution", "task_adherence",
        "relevance", "coherence", "fluency", "indirect_attack", "code_vulnerability", "ungrounded_attributes", "protected_material"
       '''
        "code_vulnerability", "ungrounded_attributes","violence", "self_harm", "sexual",
        "hate_unfairness", "protected_material"
       '''
    ],
}

# ---------------------------------------------------------------------------
# Global settings filled at startup
# ---------------------------------------------------------------------------

VAULT_URL = "https://eval-agent-kv.vault.azure.net/"

OPENAI_ENDPOINT: Optional[str] = None
OPENAI_API_KEY: Optional[str] = None
OPENAI_API_VERSION: Optional[str] = None
OPENAI_DEPLOYMENT: Optional[str] = None

AZURE_AI_PROJECT: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# Key Vault helpers
# ---------------------------------------------------------------------------

_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
_secret_client = SecretClient(vault_url=VAULT_URL, credential=_credential)


def get_secret(name: str, *, version: Optional[str] = None) -> str:
    return _secret_client.get_secret(name, version=version).value


def load_env_from_keyvault(mapping: Dict[str, str]) -> None:
    for env_var, secret_name in mapping.items():
        try:
            value = get_secret(secret_name)
            os.environ[env_var] = value
        except Exception as exc:
            print(f"Warning: could not load secret {secret_name}: {exc}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Config file '{path}' not found. Using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def merge_config(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and k in defaults and isinstance(defaults.get(k), dict):
            defaults[k] = merge_config(defaults[k], v)
        else:
            defaults[k] = v
    return defaults


def default_config() -> Dict[str, Any]:
    return {
        "project": {
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
            "project_name": "shayakproject",
            "resource_group_name": "shayak-test-rg",
            "endpoint": "https://shayak-foundry.services.ai.azure.com",
        },
        "simulators": ["direct", "indirect"],
        "evals": [
            "tool_call_accuracy",
            "intent_resolution",
            "task_adherence",
            "relevance",
            "coherence",
            "fluency",
            "code_vulnerability",
            "indirect_attack",
            "protected_material",
            "ungrounded_attributes",
        ],
        "key_vault_uri": VAULT_URL,
        "custom_prompts": [
            "Say hello and describe what you are."
        ],
        # quick_mode on by default
        "quick_mode": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2E Agent Evaluation with optional config")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Azure clients and helpers
# ---------------------------------------------------------------------------

def init_openai_from_env() -> None:
    global OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT

    load_dotenv()

    load_env_from_keyvault(
        {
            "AZURE_OPENAI_ENDPOINT": "az-openai-endpoint",
            "AZURE_OPENAI_API_KEY": "az-openai-api-key",
            "AZURE_OPENAI_API_VERSION": "az-openai-api-version",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
        }
    )

    OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

    print("AOAI endpoint:", OPENAI_ENDPOINT)
    print("AOAI deployment:", OPENAI_DEPLOYMENT)
    print("AOAI api_version:", OPENAI_API_VERSION)


def build_azure_ai_project(config: Dict[str, Any]) -> Dict[str, str]:
    return {
        "subscription_id": config["project"]["subscription_id"],
        "project_name": config["project"]["project_name"],
        "resource_group_name": config["project"]["resource_group_name"],
    }


def build_project_client(config: Dict[str, Any], credential: DefaultAzureCredential) -> AIProjectClient:
    project_name = config["project"]["project_name"]
    endpoint_base = config["project"]["endpoint"].rstrip("/")
    endpoint = f"{endpoint_base}/api/projects/{project_name}"
    return AIProjectClient(endpoint=endpoint, credential=credential)


def build_model_config() -> AzureOpenAIModelConfiguration:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAIModelConfiguration(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_deployment=OPENAI_DEPLOYMENT,
    )


def create_openai_client() -> AzureOpenAI:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# Application logic for answering user queries
# ---------------------------------------------------------------------------

def example_application_response(query: str, context: str) -> str:
    client = create_openai_client()

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
        model=OPENAI_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=10000,
    )
    message = completion.to_dict()["choices"][0]["message"]
    if isinstance(message, dict):
        message = message.get("content", "")
    return message


async def custom_simulator_callback(
    messages: Dict[str, Any],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages_list = messages["messages"]
    latest_message = messages_list[-1]

    application_input = latest_message["content"]
    context_from_message = latest_message.get("context", None)

    response = example_application_response(query=application_input, context=context_from_message)

    message = {
        "content": response,
        "role": "assistant",
        "context": context_from_message,
    }
    messages["messages"].append(message)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context_from_message,
    }


# ---------------------------------------------------------------------------
# Storage helper
# ---------------------------------------------------------------------------

def upload_to_blob(container_name: str, file_paths: List[str]) -> None:
    STORAGE_ACCOUNT_NAME = "evalsofagents"
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    for file_path in file_paths:
        blob_name = os.path.basename(file_path)
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"Uploaded: {blob_name}")


# ---------------------------------------------------------------------------
# Prompt generation via simulators
# ---------------------------------------------------------------------------

def extract_queries_from_outputs(
    outputs: Any,
    records: List[PromptRecord],
    simulator: str,
    scenario: Optional[str],
) -> None:
    json_lines = outputs.to_eval_qr_json_lines().splitlines()
    for line in json_lines:
        try:
            obj = json.loads(line)
            q = obj.get("query")
            if q:
                records.append(PromptRecord(prompt=q, simulator=simulator, scenario=scenario))
        except json.JSONDecodeError:
            print(f"Could not parse JSON line from simulator: {line}")


async def run_simulators(config: Dict[str, Any], credential: DefaultAzureCredential) -> List[PromptRecord]:
    if AZURE_AI_PROJECT is None:
        raise RuntimeError("AZURE_AI_PROJECT is not initialized")

    records: List[PromptRecord] = []

    simulators_to_run = [s.lower() for s in config.get("simulators", [])]

    quick_mode = config.get("quick_mode", False)
    quick_mode = False
    max_results = 1 if quick_mode else 20
    max_turns = 10 if quick_mode else 3

    custom_simulator = AdversarialSimulator(credential=credential, azure_ai_project=AZURE_AI_PROJECT)
    direct_attack_simulator = DirectAttackSimulator(azure_ai_project=AZURE_AI_PROJECT, credential=credential)
    indirect_attack_simulator = IndirectAttackSimulator(azure_ai_project=AZURE_AI_PROJECT, credential=credential)

    if "adversarial" in simulators_to_run:
        if quick_mode:
            scenarios = [AdversarialScenario.ADVERSARIAL_QA]
        else:
            scenarios = [
                AdversarialScenario.ADVERSARIAL_QA,
                AdversarialScenario.ADVERSARIAL_CONVERSATION,
                AdversarialScenario.ADVERSARIAL_SUMMARIZATION,
                AdversarialScenario.ADVERSARIAL_SEARCH,
                AdversarialScenario.ADVERSARIAL_REWRITE,
                AdversarialScenario.ADVERSARIAL_CONTENT_GEN_UNGROUNDED,
                AdversarialScenario.ADVERSARIAL_CONTENT_PROTECTED_MATERIAL,
            ]

        for scenario in scenarios:
            try:
                outputs = await custom_simulator(
                    scenario=scenario,
                    max_simulation_results=max_results,
                    target=custom_simulator_callback,
                )
                extract_queries_from_outputs(
                    outputs,
                    records,
                    simulator="adversarial",
                    scenario=scenario.name if hasattr(scenario, "name") else str(scenario),
                )
            except Exception as exc:
                print(f"Adversarial simulator failed for scenario {scenario}: {exc}")

    if "direct" in simulators_to_run:
        try:
            outputs = await direct_attack_simulator(
                scenario=AdversarialScenario.ADVERSARIAL_CONVERSATION,
                max_simulation_results=max_results,
                target=custom_simulator_callback,
            )
            extract_queries_from_outputs(
                outputs,
                records,
                simulator="direct",
                scenario=AdversarialScenario.ADVERSARIAL_CONVERSATION.name,
            )
            print("Direct attack prompts collected")
        except Exception as exc:
            print(f"Direct attack simulator failed: {exc}")

    if "indirect" in simulators_to_run:
        try:
            outputs = await indirect_attack_simulator(
                max_simulation_results=max_results,
                target=custom_simulator_callback,
                max_conversation_turns=max_turns,
            )
            extract_queries_from_outputs(
                outputs,
                records,
                simulator="indirect",
                scenario="indirect_attack",
            )
            print("Indirect attack prompts collected")
        except Exception as exc:
            print(f"Indirect attack simulator failed: {exc}")

    return records


# ---------------------------------------------------------------------------
# Evaluation setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Top level driver
# ---------------------------------------------------------------------------

def main() -> None:
    global AZURE_AI_PROJECT

    from pprint import pprint

    args = parse_args()
    config = merge_config(default_config(), load_config(args.config))

    file_suffix = uuid.uuid4().hex
    print("Run GUID:", file_suffix)

    prompts_file = f"list_of_prompts_{file_suffix}.txt"
    all_pairs_path = f"query_response_pairs_{file_suffix}.jsonl"
    all_evals_path = f"eval_results_{file_suffix}.txt"
    evaluation_data_file = "evaluationDataAdversarialData.jsonl"

    init_openai_from_env()

    AZURE_AI_PROJECT = build_azure_ai_project(config)
    print("Azure AI project:", AZURE_AI_PROJECT)

    credential = DefaultAzureCredential()

    project_client = build_project_client(config, credential)
    '''
    The below are validator and Discovery of workloads agents
    #agent = project_client.agents.get_agent(agent_id="asst_40jxCEVxQniq4Pr7lDxTxeYu")
    agent = project_client.agents.get_agent(agent_id="asst_FJFfBZAkysYF8LwvbU49hEra") 
    '''
    # Troubleshooting AI foundry agent in MSFT and AI hub service deployments below
    agent = project_client.agents.get_agent(agent_id="asst_OmtWFZGuXJXSfiJ7C41fHDk6")
    #agent = project_client.agents.get_agent(agent_id="asst_cUo5n03tG9VhuAY4wsQIL07c") 
    print(f"Fetched agent, ID: {agent.id}")

    prompt_records = asyncio.run(run_simulators(config, credential))

    # Add custom prompts to the same list, tagged with simulator "custom"
    custom_prompts = config.get("custom_prompts", [])
    for p in custom_prompts:
        prompt_records.append(PromptRecord(prompt=p, simulator="custom", scenario=None))

    print("Final list of prompts (with simulator tags):")
    pprint(prompt_records, width=200)

    # Save prompt records as JSON so simulator info is preserved
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(
            [pr.__dict__ for pr in prompt_records],
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        print("Uploading list of prompts information to azure storage")
        upload_to_blob(container_name="list-of-prompts-1", file_paths=[prompts_file])
    except Exception as exc:
        print("Exception in uploading list of prompts")
        print(exc)

    model_config = build_model_config()
    evaluator_map = build_evaluators(model_config=model_config, credential=credential)
    enabled_evals = config.get("evals", [])

    process_prompts_with_agent(
        prompt_records=prompt_records,
        project_client=project_client,
        agent_id=agent.id,
        evaluator_map=evaluator_map,
        enabled_evals=enabled_evals,
        all_pairs_path=all_pairs_path,
        evaluation_data_file=evaluation_data_file,
        run_guid=file_suffix,
    )

    active_evaluators = {k: v for k, v in evaluator_map.items() if k in enabled_evals}

    try:
        response = evaluate(
            data=evaluation_data_file,
            evaluators=active_evaluators,
            azure_ai_project="https://padmajat-agenticai-hack-resource.services.ai.azure.com/api/projects/padmajat-agenticai-hackathon25",
        )
    except Exception as exc:
        print("Batch evaluate failed:")
        print(exc)
        response = {}

    pprint([pr.prompt for pr in prompt_records], width=200)
    if isinstance(response, dict):
        pprint(f"Azure ML Studio URL: {response.get('studio_url')}")
    pprint(response)

    try:
        with open(all_evals_path, "a", encoding="utf-8") as evals_handle:
            evals_handle.write(str(response) + "\n")
    except Exception as write_err:
        print("Failed to append evals to txt file:", write_err)

    try:
        print("Uploading the query-response pairs to the storage account")
        upload_to_blob(container_name="query-response-pairs-1", file_paths=[all_pairs_path])
        print("Uploaded query json pairs results")
    except Exception as exc:
        print("Exception uploading query response pairs")
        print(exc)

    try:
        print("Uploading all evals for this run to azure storage")
        upload_to_blob(container_name="list-of-evals-1", file_paths=[all_evals_path])
    except Exception as exc:
        print("Exception in uploading all evals")
        print(exc)


if __name__ == "__main__":
    main()
