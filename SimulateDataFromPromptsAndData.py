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

import json
from pathlib import Path

model_config = {
    "azure_endpoint": azure_openai_endpoint,
    "azure_deployment": azure_openai_deployment,
    "api_version": azure_openai_api_version,
}

from azure.ai.evaluation.simulator import Simulator

simulator = Simulator(model_config=model_config)

from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def call_to_your_ai_application(query: str) -> str:
    # logic to call your application
    # use a try except block to catch any errors
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_ad_token_provider=token_provider,
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    message = completion.to_dict()["choices"][0]["message"]
    # change this to return the response from your application
    return message["content"]


async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,  # noqa: ANN401
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    query = latest_message["content"]
    context = None
    # call your endpoint or ai application here
    response = call_to_your_ai_application(query)
    # we are formatting the response to follow the openAI chat protocol format
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {
            "citations": None,
        },
    }
    messages["messages"].append(formatted_response)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

import wikipedia

wiki_search_term = "Leonardo da vinci"
wiki_title = wikipedia.search(wiki_search_term)[0]
wiki_page = wikipedia.page(wiki_title)
text = wiki_page.summary[:5000]

outputs = await simulator(
    target=callback,
    text=text,
    num_queries=4,
    max_conversation_turns=3,
    tasks=[
        f"I am a student and I want to learn more about {wiki_search_term}",
        f"I am a teacher and I want to teach my students about {wiki_search_term}",
        f"I am a researcher and I want to do a detailed research on {wiki_search_term}",
        f"I am a statistician and I want to do a detailed table of factual data concerning {wiki_search_term}",
    ],
)

current_directory = Path.cwd()
user_override_prompty = Path(current_directory) / "user_override.prompty"
user_prompty_kwargs = {"mood": "happy"}

outputs = await simulator(
    target=callback,
    text=text,
    num_queries=4,
    max_conversation_turns=1,
    tasks=[
        f"I am a student and I want to learn more about {wiki_search_term}",
        f"I am a teacher and I want to teach my students about {wiki_search_term}",
        f"I am a researcher and I want to do a detailed research on {wiki_search_term}",
        f"I am a statistician and I want to do a detailed table of factual data concerning {wiki_search_term}",
    ],
    user_simulator_prompty=user_override_prompty,
    user_simulator_prompty_options=user_prompty_kwargs,
)

from pathlib import Path

output_file = Path("output.json")
with output_file.open("a") as f:
    json.dump(outputs, f)

eval_input_data_json_lines = ""
for output in outputs:
    query = None
    response = None
    context = text
    ground_truth = text
    for message in output["messages"]:
        if message["role"] == "user":
            query = message["content"]
        if message["role"] == "assistant":
            response = message["content"]
    if query and response:
        eval_input_data_json_lines += (
            json.dumps(
                {
                    "query": query,
                    "response": response,
                    "context": context,
                    "ground_truth": ground_truth,
                }
            )
            + "\n"
        )

eval_input_data_file = Path("eval_input_data.jsonl")
with eval_input_data_file.open("w") as f:
    f.write(eval_input_data_json_lines)

from azure.ai.evaluation import evaluate, QAEvaluator

qa_evaluator = QAEvaluator(model_config=model_config)

eval_output = evaluate(
    data=str(eval_input_data_file),
    evaluators={"QAEvaluator": qa_evaluator},
    evaluator_config={
        "QAEvaluator": {
            "column_mapping": {
                "query": "${data.query}",
                "response": "${data.response}",
                "context": "${data.context}",
                "ground_truth": "${data.ground_truth}",
            }
        }
    },
    azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
    output_path="./myevalresults.json",  # optional to store the evaluation results in a file
)

pprint(eval_output)
