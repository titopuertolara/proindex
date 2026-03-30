# Bedrock PageIndex: Vectorless, Reasoning-based RAG with AWS Bedrock

A port of [PageIndex](https://github.com/VectifyAI/PageIndex) that uses **AWS Bedrock** instead of OpenAI/LiteLLM for all LLM calls.

PageIndex is a **vectorless**, **reasoning-based RAG** system that builds a **hierarchical tree index** from long documents and uses LLMs to **reason** over that index for agentic, context-aware retrieval — no vector database, no chunking.

## Key Differences from Original PageIndex

| Feature | Original PageIndex | Bedrock PageIndex |
|---|---|---|
| LLM Provider | OpenAI / LiteLLM | AWS Bedrock (Converse API) |
| Token Counting | LiteLLM | tiktoken |
| Agentic Demo | OpenAI Agents SDK | Bedrock Converse tool-use loop |
| Default Model | gpt-4o | Claude Sonnet 4 (via Bedrock) |
| Config | `OPENAI_API_KEY` | AWS credentials |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials

```bash
# Option A: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Option B: AWS CLI profile
aws configure

# Option C: IAM role (if running on EC2/Lambda/ECS)
# No configuration needed
```

### 3. Generate PageIndex structure for your PDF

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```

### 4. Run the Agentic Vectorless RAG Demo

```bash
python3 examples/agentic_vectorless_rag_demo.py
```

## Configuration

Edit `pageindex/config.yaml` to change defaults:

```yaml
model: "us.anthropic.claude-sonnet-4-20250514-v1:0"
retrieve_model: "us.anthropic.claude-sonnet-4-20250514-v1:0"
aws_region: "us-east-1"
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

### Supported Bedrock Models

Any model available via the [Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html):

- `us.anthropic.claude-sonnet-4-20250514-v1:0` (default)
- `us.anthropic.claude-opus-4-20250514-v1:0`
- `us.amazon.nova-pro-v1:0`
- `us.amazon.nova-lite-v1:0`
- `us.meta.llama3-3-70b-instruct-v1:0`

## Architecture

```
PDF Document
    │
    ▼
┌──────────────────────┐
│  PageIndex Builder    │  Uses Bedrock Converse API to:
│  (page_index.py)      │  1. Detect TOC
│                       │  2. Extract hierarchical structure
│                       │  3. Verify & fix page mappings
│                       │  4. Generate node summaries
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Tree Structure JSON  │  Hierarchical index of document
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Agentic RAG Agent    │  Bedrock Converse tool-use loop:
│  (Converse API)       │  - get_document()
│                       │  - get_document_structure()
│                       │  - get_page_content(pages)
└──────────────────────┘
```

## CLI Options

```
--pdf_path              Path to PDF file
--md_path               Path to Markdown file
--model                 Bedrock model ID (overrides config.yaml)
--toc-check-pages       Pages to check for TOC (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node ID (yes/no, default: yes)
--if-add-node-summary   Add node summary (yes/no, default: yes)
--if-add-doc-description Add doc description (yes/no, default: no)
--if-add-node-text      Add full text to nodes (yes/no, default: no)
```

## Credits

Based on [PageIndex by VectifyAI](https://github.com/VectifyAI/PageIndex). This is a community port for AWS Bedrock — not affiliated with Vectify AI.
