# Jupyter notebook Samples for Tanzu Platform for Cloud Foundry
This repository provides JupyterLab notebooks built with Python, leveraging LangChain and LangGraph. They demonstrate patterns for connecting to a variety of services—such as large language models, data services, and vector databases while incorporating service discovery and credential management with Tanzu Platform for Cloud Foundry.

## Why JupyterLab on TPCF?

### Collaborative Research Platform
- Data Scientists can prototype AI models and analyze results interactively
- ML Engineers can test model integrations and performance
- Application Developers can experiment with AI features before building production apps
- Platform Teams maintain governance and cost control across all usage
### Integrated Workflow Benefits
When JupyterLab is bound to multiple services, you get:
- Data Services connectivity (like Postgres, Valkey, Gemfire, RabbitMQ) for data exploration and feature engineering
- Pre-approved LLM access for natural language processing and generation experiments
- Vector database capabilities (Postgres or Gemfire) for embedding storage and similarity search
- Automatic service discovery and credential management through Tanzu Platform for Cloud Foundry
- Built-in observability for tracking resource usage and model performance


## Getting started
- Deploy JupyterLab for Cloud Foundry into your TPCF environment using the repository https://github.com/yannicklevederpvtl/cf-jupyterlab-uv
- git clone the cf-jupyterlab-samples and start coding
  





