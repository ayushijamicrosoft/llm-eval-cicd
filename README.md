# Evaluation of Agents


In this project, we utilize **Azure AI Foundry Python SDK** to evaluate AI agents on various parameters like Groundedness, Relevance, Safety, Tool Call Adherance.

There are three main stages: 

1. Generation of synthetic prompts for different scenarios like direct/indirect jail break attacks, advarsarial prompts - using Simulator. User can add their own custom prompts to this list as well. 
2. Execution of multi-turn conversations with prompts to the agent - every conversation is called as "thread"
3. Convertion of every conversation to the correct format (using AI Agent converter function) for evaluation with different evaluators.


## Main files to run
1. **EndtoEndAgentEvaluation.py** - for end to end flow
2. Workflow to run the python file: **E2EAgentEvaluation.yml**
   
## How to run the repository?

## Prerequisites (Highly Important ‼️‼️‼️)

1. Creation of Azure ML Studio with same name as your AI Foundry project in same RG is essential!
2. The client id which is being used (in repo secrets) should have "Contributor" permissions to your Azure ML Studio workspace and AI foundry project.Additionally, at least Azure Storage Blob Contributor permissions to the storage account. .
3. Check the region of the project - some simulators and evaluators may or may not be supported in that region. 

## Initial Setup
1. Installation of python libraries: **requirements.txt**
2. Open AI - Key, Subscription, Client ID etc: Repository Secrets - can be updated there.
3. Workflows results in Github Actions tab
4. Azure ML Studio resource required for resources.

### Create an entra id app. 
1. Navigate to the portal, and click on **New registration**.
2. Navigate to certificate and secrets and in federated credentials -> add credential. Choose github, and main branch
3. Grant RBAC permissions to this app to the ai foundry resource and mentioned OIDC Subscription.


### Workflows
1. GitHub Actions - are used to run the python files, everytime a change is pushed to the main branch. Refer to the .workflows/ folder for the yml file.
2. The repository secrets are stored in Github settings > Secrets
3. Output of every workflow is visible in **Actions** tab.
4. You can download the final output file from the workflow run.
5. The final output file is in the form of *.jsonl.
6. Additionally, the logs are visible in Azure ML Studio link in Azure Portal.

### Use the output file
1. Go to Azure ML Studio > Evaluations tab > Logs + Metrics
2. Pick up the **instance_results.jsonl** file and copy paste the contents into a **instance_results.json** file. 
3. Locally, convert this **instance_results.json** file into **converted_data.jsonl** file by running the python file **format_output.py** from this GitHub Repo.
4. Now, upload this **converted_data.jsonl** file to the Azure AI Foundry > Evaluation > Create New Evaluation.
5. In Azure AI Studio Evals - you can choose which evals you want to reflect.Additionally, Azure ML studio has those same details as well.

### List of Files to use. 
1. To run the end to end flow: **EndtoEndAgentEvaluation.py**
2. To convert the logs in Azure ML Studio to Azure AI Evals results: **format_output.py**
3. To run the workflows: **.github/.workflows/**

#### Experimental Files:
Files present in the Experimental-Files/* folder

1. Check only the format of thread to be passed into the evaluator: **tool-call-accuracy.py**
2. Pick up all existing threads of agent and pass them into the evaluator (NO DATA SIMULATED): **tool-call-accuracy-automated.py**
3. Sample tool call functions: **user_functions.py**
4. Generate synthetic data and evaluation with model for groundedness (NO Agents evaluation involved): **groundedness-evals.py**
5. Generate synthetic data and evaluation with model for content safety (NO Agents evaluation involved):**content-safety.py**
6. 

### Sample formats
Files present in the Sample-Formats/* folder

1. Format for uploading evaluation to Azure AI foundry: **evaluation_data.jsonl**, **llm-evals.jsonl**
2. Format for uploading data for groundedness: **grounding.json**

### Other Attempts
Files present in the Other-Attempts/* folder

1. Evaluate agents on existing conversations: **agent-eval-existing-conversations.py**
2. Simulate the conversation with agent, and evaluate it (whole flow) - with all the evals: **agent-evaluation-all-evaluators.py**

### PS:

1. The python file runs only for the GitHub Actions, and would not run locally - this is because of a setup on Azure.
2. Azure Foundry Evals is currently in Preview stage - so there could be more changes, but the idea of the project is to run an overall flow.

