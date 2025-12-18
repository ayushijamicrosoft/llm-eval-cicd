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
from azure.storage.blob import BlobClient
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


VAULT_URL = "https://eval-agent-kv.vault.azure.net/"
AZURE_AI_PROJECT = None

# --------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------

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
        "agent_id": "asst_OmtWFZGuXJXSfiJ7C41fHDk6",
        "storage_connection_string": "DefaultEndpointsProtocol=https;AccountName=evalsofagents;AccountKey=1zVYXqPzCUVTRVcROPypVju8FVcKTX9hHpLIJVfg9w6vxwsmdDanWz+lqj7UI+cDTntKyJrfaEvP+AStDxM2Yg==;EndpointSuffix=core.windows.net",
        "storage_container": "list-of-thread-ids",
        "storage_blob": "thread_ids_run_2c844df6f7a64a5f93c65f0040726045.txt",
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
            #"protected_material",
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
    parser.add_argument("--path_to_thread_ids", type=str, required=True, default=None, help="Path to the file in storage account containing thread ids for evaluation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Azure clients and helpers
# ---------------------------------------------------------------------------

def init_openai_from_env() -> None:
    global OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION, OPENAI_DEPLOYMENT, OPENAI_ENDPOINT_EVALS, OPENAI_API_KEY_EVALS, OPENAI_API_VERSION_EVALS, OPENAI_DEPLOYMENT_EVALS

    load_dotenv()

    load_env_from_keyvault(
        {
            "AZURE_OPENAI_ENDPOINT": "az-openai-endpoint",
            "AZURE_OPENAI_API_KEY": "az-openai-api-key",
            "AZURE_OPENAI_API_VERSION": "az-openai-api-version",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "az-openai-deploy",
            "AZURE_OPENAI_ENDPOINT_EVALS": "az-openai-endpoint-evals",
            "AZURE_OPENAI_API_KEY_EVALS": "az-openai-api-key-evals",
            "AZURE_OPENAI_API_VERSION_EVALS": "az-openai-api-version-evals",
            "AZURE_OPENAI_CHAT_DEPLOYMENT_EVALS": "az-openai-deploy-evals"
        }
    )

    OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
    OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

    OPENAI_ENDPOINT_EVALS = os.environ.get("AZURE_OPENAI_ENDPOINT_EVALS")
    OPENAI_API_KEY_EVALS = os.environ.get("AZURE_OPENAI_API_KEY_EVALS")
    OPENAI_API_VERSION_EVALS = os.environ.get("AZURE_OPENAI_API_VERSION_EVALS")
    OPENAI_DEPLOYMENT_EVALS = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_EVALS")
    
    print("AOAI endpoint:", OPENAI_ENDPOINT)
    print("AOAI deployment:", OPENAI_DEPLOYMENT)
    print("AOAI api_version:", OPENAI_API_VERSION)

    print("AOAI evals endpoint:", OPENAI_ENDPOINT_EVALS)
    print("AOAI evals deployment:", OPENAI_DEPLOYMENT_EVALS)
    print("AOAI evals api_version:", OPENAI_API_VERSION_EVALS)


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
    if not all([OPENAI_ENDPOINT_EVALS, OPENAI_API_KEY_EVALS, OPENAI_API_VERSION_EVALS, OPENAI_DEPLOYMENT_EVALS]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAIModelConfiguration(
        azure_endpoint=OPENAI_ENDPOINT_EVALS,
        api_key=OPENAI_API_KEY_EVALS,
        api_version=OPENAI_API_VERSION_EVALS,
        azure_deployment=OPENAI_DEPLOYMENT_EVALS,
    )


def create_openai_client() -> AzureOpenAI:
    if not all([OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION]):
        raise RuntimeError("OpenAI settings are not fully initialized.")
    return AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    )


def is_updated_agents():
    """Flag for whether updated agents API needs to be used."""
    return Version(projects_version) > Version("1.0.0b10") or projects_version.startswith(
        "1.0.0a"
    )


# --------------------------------------------------------------------
# Modular functions for Azure AI Project + Agent
# --------------------------------------------------------------------


def get_project_client(project_endpoint: str, credential=None) -> AIProjectClient:
    """
    Create and return an AIProjectClient.
    """
    if credential is None:
        credential = DefaultAzureCredential()

    client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )
    return client


def get_agent(project_client: AIProjectClient, agent_id: str):
    """
    Fetch and return an agent given its id.
    """
    agent = project_client.agents.get_agent(agent_id=agent_id)
    print(f"Fetched agent, ID: {agent.id}")
    return agent


# --------------------------------------------------------------------
# Fetch thread ids from Azure Blob Storage
# --------------------------------------------------------------------


def get_thread_ids_from_blob(
    connection_string: str, container_name: str, blob_name: str
) -> list[str]:
    """
    Read a txt file from Azure Blob Storage and return a list of thread ids.
    Assumes:
      - One thread id per line in the blob.
    """
    if not all([connection_string, container_name, blob_name]):
        raise ValueError(
            "Storage connection string, container, and blob name must all be set."
        )

    blob_client = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name,
    )

    download_stream = blob_client.download_blob()
    content = download_stream.readall().decode("utf-8")

    thread_ids = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]
    print(f"Fetched {len(thread_ids)} thread ids from blob {container_name}/{blob_name}")
    return thread_ids


# --------------------------------------------------------------------
# Prepare evaluation data
# --------------------------------------------------------------------


def prepare_evaluation_data_file(
    project_client: AIProjectClient,
    thread_ids: list[str],
    output_file: str = "freshEvaluationData.jsonl",
) -> str:
    """
    Use AIAgentConverter to prepare evaluation data jsonl from thread ids.
    """
    converter = AIAgentConverter(project_client)
    evaluation_data = converter.prepare_evaluation_data(
        thread_ids=thread_ids,
        filename=output_file,
    )
    print(f"Wrote evaluation data for {len(thread_ids)} threads to {output_file}")
    return output_file


# --------------------------------------------------------------------
# Set up evaluators and run evaluation
# --------------------------------------------------------------------

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
    #protected_material = ProtectedMaterialEvaluator(**content_safety_kwargs)
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
        #"protected_material": protected_material,
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


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

import os
import time
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    EvaluationRule,
    ContinuousEvaluationRuleAction,
    EvaluationRuleFilter,
    EvaluationRuleEventType,
)

load_dotenv()


AZURE_AI_PROJECT = None

def main():
    global AZURE_AI_PROJECT
    args = parse_args()
    config = merge_config(default_config(), load_config(args.config))
    print(args.path_to_thread_ids)
    config["storage_blob"] = args.path_to_thread_ids
    credential = DefaultAzureCredential()
    
    init_openai_from_env()
    
    AZURE_AI_PROJECT = build_azure_ai_project(config)
    print("Azure AI project:", AZURE_AI_PROJECT)
    
    credential = DefaultAzureCredential()
    
    project_client = build_project_client(config, credential)
    os.environ["AZURE_AI_AGENT_NAME"] = "contevals"
    os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"] = "gpt-4.1"
    
    with project_client.get_openai_client() as openai_client:
        agent = project_client.agents.create_version(
            agent_name=os.environ["AZURE_AI_AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
                instructions="You are a helpful assistant that answers general questions",
            ),
        )
        print(f"Agent created (id: {agent.id}, name: {agent.name}, version: {agent.version})")
    
        # Setup agent continuous evaluation
    
        data_source_config = {"type": "azure_ai_source", "scenario": "responses"}
        testing_criteria = [
            {"type": "azure_ai_evaluator", "name": "violence_detection", "evaluator_name": "builtin.violence"}
        ]
        eval_object = openai_client.evals.create(
            name="Continuous Evaluation",
            data_source_config=data_source_config,  # type: ignore
            testing_criteria=testing_criteria,  # type: ignore
        )
        print(f"Evaluation created (id: {eval_object.id}, name: {eval_object.name})")
    
        continuous_eval_rule = project_client.evaluation_rules.create_or_update(
            id="my-continuous-eval-rule",
            evaluation_rule=EvaluationRule(
                display_name="My Continuous Eval Rule",
                description="An eval rule that runs on agent response completions",
                action=ContinuousEvaluationRuleAction(eval_id=eval_object.id, max_hourly_runs=100),
                event_type=EvaluationRuleEventType.RESPONSE_COMPLETED,
                filter=EvaluationRuleFilter(agent_name=agent.name),
                enabled=True,
            ),
        )
        print(
            f"Continuous Evaluation Rule created (id: {continuous_eval_rule.id}, name: {continuous_eval_rule.display_name})"
        )
    
        # Run agent
    
        conversation = openai_client.conversations.create(
            items=[{"type": "message", "role": "user", "content": "What is the size of France in square miles?"}],
        )
        print(f"Created conversation with initial user message (id: {conversation.id})")
    
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
            input="",
        )
        print(f"Response output: {response.output_text}")
    
        # Loop for 5 questions
    
        MAX_QUESTIONS = 10
        for i in range(0, MAX_QUESTIONS):
            openai_client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user", "content": f"Question {i}: What is the capital city?"}],
            )
            print(f"Added a user message to the conversation")
    
            response = openai_client.responses.create(
                conversation=conversation.id,
                extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
                input="",
            )
            print(f"Response output: {response.output_text}")
    
            # Wait for 10 seconds for evaluation, and then retrieve eval results
    
            time.sleep(10)
            eval_run_list = openai_client.evals.runs.list(
                eval_id=eval_object.id,
                order="desc",
                limit=10,
            )
    
            if len(eval_run_list.data) > 0:
                eval_run_ids = [eval_run.id for eval_run in eval_run_list.data]
                print(f"Finished evals: {' '.join(eval_run_ids)}")
    
        # Get the report_url
    
        print("Agent runs finished")
    
        MAX_LOOP = 20
        for _ in range(0, MAX_LOOP):
            print(f"Waiting for eval run to complete...")
    
            eval_run_list = openai_client.evals.runs.list(
                eval_id=eval_object.id,
                order="desc",
                limit=10,
            )
    
            if len(eval_run_list.data) > 0 and eval_run_list.data[0].report_url:
                run_report_url = eval_run_list.data[0].report_url
                # Remove the last 2 URL path segments (run/continuousevalrun_xxx)
                report_url = '/'.join(run_report_url.split('/')[:-2])
                print(f"To check evaluation runs, please open {report_url} from the browser")
                break
    
            time.sleep(10)
    args = parse_args()
    config = merge_config(default_config(), load_config(args.config))
    print(args.path_to_thread_ids)
    config["storage_blob"] = args.path_to_thread_ids
    credential = DefaultAzureCredential()

    init_openai_from_env()

    AZURE_AI_PROJECT = build_azure_ai_project(config)
    print("Azure AI project:", AZURE_AI_PROJECT)

    credential = DefaultAzureCredential()
    
    project_client = build_project_client(config, credential)
    agent = project_client.agents.get_agent(agent_id=config["agent_id"])
    # 3. Fetch thread ids from blob txt file
    thread_ids = get_thread_ids_from_blob(
        connection_string=config["storage_connection_string"],
        container_name=config["storage_container"],
        blob_name=config["storage_blob"],
    )

    model_config = build_model_config()
    evaluators={
        "Relevance": {"Id": EvaluatorIds.Relevance.value},
        "Fluency": {"Id": EvaluatorIds.Fluency.value},
        "Coherence": {"Id": EvaluatorIds.Coherence.value},
      },

    for thread in thread_ids:
        '''
        messages = list(project_client.agents.messages.list(thread))
        run_id = messages[0].run_id
        thread_id = thread
        '''
        openai_client = project_client.get_openai_client(api_version="2024-10-21");
        data_source_config = {"type": "azure_ai_source", "scenario": "responses"}
        testing_criteria = [
            {"type": "azure_ai_evaluator", "name": "violence_detection", "evaluator_name": "builtin.violence"}
        ]
        

        eval_object = openai_client.evals.create(
            name="cont-evals",
            data_source_config=data_source_config,  # type: ignore
            testing_criteria=testing_criteria,  # type: ignore
        )
    
        continuous_eval_rule = project_client.evaluation_rules.create_or_update(
        id="my-continuous-eval-rule",
        evaluation_rule=EvaluationRule(
            display_name="My Continuous Eval Rule",
            description="An eval rule that runs on agent response completions",
            action=ContinuousEvaluationRuleAction(eval_id=eval_object.id, max_hourly_runs=100),
            event_type=EvaluationRuleEventType.RESPONSE_COMPLETED,
            filter=EvaluationRuleFilter(agent_name=agent.name),
            enabled=True,
            ),
        )

        MAX_LOOP = 20
        for _ in range(0, MAX_LOOP):
            print(f"Waiting for eval run to complete...")
        
            eval_run_list = openai_client.evals.runs.list(
                eval_id=eval_object.id,
                order="desc",
                limit=10,
            )
        
            if len(eval_run_list.data) > 0 and eval_run_list.data[0].report_url:
                run_report_url = eval_run_list.data[0].report_url
                # Remove the last 2 URL path segments (run/continuousevalrun_xxx)
                report_url = '/'.join(run_report_url.split('/')[:-2])
                print(f"To check evaluation runs, please open {report_url} from the browser")
                break
        
            time.sleep(10)

    
if __name__ == "__main__":
    main()
