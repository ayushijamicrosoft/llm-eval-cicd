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

# Initialize the environment variables for Azure OpenAI and the AI project details
openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
model_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
os.environ["AZURE_DEPLOYMENT_NAME"] = deployment
os.environ["AZURE_API_VERSION"] = api_version
os.environ["AZURE_IDENTITY_ENABLE_INTERACTIVE"] = "1"

azure_ai_project = {
    "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
    "project_name": "padmajat-agenticai-hackathon25",
    "resource_group_name": "rg-padmajat-2824",
}

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
    deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

    # Get a client handle for the AOAI model
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_API_VERSION"),
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
            ToolOutput,
         model=deployment,
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
        azure_ai_project={
            "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
                  "project_name": "padmajat-agenticai-hackathon25",
                  "resource_group_name": "rg-padmajat-2824",
        }
        
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
        
        
        tool_call_accuracy(query=converted_data['query'], response=converted_data['response'], tool_definitions=converted_data['tool_definitions'])
        intent_resolution(query=converted_data['query'], response=converted_data['response'])
        task_adherence(query=converted_data['query'], response=converted_data['response'])
        violence(query=converted_data['query'], response=converted_data['response'])
        relevance(query=converted_data['query'], response=converted_data['response'])
        coherence(query=converted_data['query'], response=converted_data['response'])
        fluency(response=converted_data['response'])
        self_harm(query=converted_data['query'], response=converted_data['response'])
        sexual(query=converted_data['query'], response=converted_data['response'])
        hate_unfairness(query=converted_data['query'], response=converted_data['response'])
        code_vulnerability(query=converted_data['query'], response=converted_data['response'])
        indirect_attack(query=converted_data['query'], response=converted_data['response'])
        protected_material(query=converted_data['query'], response=converted_data['response'])
        
        
        from azure.ai.evaluation import evaluate
        
        response = evaluate(
            data=file_name,
            evaluators={
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
               "protected_material": protected_material
            },
            azure_ai_project={
                "subscription_id": "49d64d54-e966-4c46-a868-1999802b762c",
                  "project_name": "padmajat-agenticai-hackathon25",
                  "resource_group_name": "rg-padmajat-2824",
            }
        )
        from pprint import pprint
    
        pprint(list_of_prompts, width=200)
        pprint(f'Azure ML Studio URL: {response.get("studio_url")}')
        pprint(response)
        
        # Save evaluation response to JSON file
        with open("metrics.json", "w") as f:
            json.dump(response, f, indent=2, default=str)
    
    except Exception as exception:
        print("exception occured!")
        print(exception)
        continue;
