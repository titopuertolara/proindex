"""
Agentic Vectorless RAG with PageIndex - AWS Bedrock Demo

A simple example of building a document QA agent with self-hosted PageIndex
and the AWS Bedrock Converse API (tool-use). Instead of vector similarity
search and chunking, PageIndex builds a hierarchical tree index and uses
agentic LLM reasoning for human-like, context-aware retrieval.

Agent tools:
  - get_document()           — document metadata (status, page count, etc.)
  - get_document_structure() — tree structure index of a document
  - get_page_content()       — retrieve text content of specific pages

Steps:
  1 — Index a PDF and view its tree structure index
  2 — View document metadata
  3 — Ask a question (Bedrock agent reasons over the index and auto-calls tools)

Requirements:
  pip install -r requirements.txt
  AWS credentials configured (via env vars, ~/.aws/credentials, or IAM role)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex import PageIndexClient
import pageindex.utils as utils

_EXAMPLES_DIR = Path(__file__).parent
DOC_PATH = _EXAMPLES_DIR / "documents" / "BILLS-119hr7148enr.md"
WORKSPACE = _EXAMPLES_DIR / "workspace"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant powered by AWS Bedrock.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to get the top-level outline (depth=1). Each node has a node_id and may have a child_count.
- Call get_section_detail(node_id="0015") to drill into a section's children. Use depth to control how many levels deep.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document. For Markdown documents, use line numbers from the structure's line_num field.
- Navigate the tree: structure first (top-level), then drill into relevant sections, then fetch page content.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""

# Bedrock tool specifications for Converse API
TOOL_SPECS = [
    {
        "name": "get_document",
        "description": "Get document metadata: status, page count, name, and description.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        },
    },
    {
        "name": "get_document_structure",
        "description": "Get the document's tree structure outline. Returns top-level nodes with node_id, title, summary, and child_count. Use this first to understand the document layout, then drill into sections with get_section_detail.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "How many levels deep to return. Default 1 (top-level only). Use 2 to also see immediate children.",
                    }
                },
                "required": [],
            }
        },
    },
    {
        "name": "get_section_detail",
        "description": "Get the subtree of a specific section by node_id. Returns the node and its children. Use this to drill into a section found via get_document_structure.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node_id to drill into, e.g. '0015'",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many levels of children to include. Default 1.",
                    },
                },
                "required": ["node_id"],
            }
        },
    },
    {
        "name": "get_page_content",
        "description": "Get the text content of specific pages or line numbers. Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12. For Markdown documents, use line numbers from the structure's line_num field.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "string",
                        "description": "Page range string, e.g. '5-7', '3,8', or '12'",
                    }
                },
                "required": ["pages"],
            }
        },
    },
]


def query_agent(
    client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False
) -> str:
    """Run a document QA agent using AWS Bedrock Converse with tool-use.

    Prints tool calls and the final answer, returns the full answer string.
    """

    # Build tool handlers that close over client and doc_id
    def handle_get_document(input_dict):
        return client.get_document(doc_id)

    def handle_get_document_structure(input_dict):
        depth = input_dict.get("depth", 1)
        return client.get_document_structure(doc_id, depth=depth)

    def handle_get_section_detail(input_dict):
        node_id = input_dict.get("node_id")
        depth = input_dict.get("depth", 1)
        return client.get_section_detail(doc_id, node_id, depth=depth)

    def handle_get_page_content(input_dict):
        pages = input_dict.get("pages", "1")
        return client.get_page_content(doc_id, pages)

    tool_handlers = {
        "get_document": handle_get_document,
        "get_document_structure": handle_get_document_structure,
        "get_section_detail": handle_get_section_detail,
        "get_page_content": handle_get_page_content,
    }

    system = [{"text": AGENT_SYSTEM_PROMPT}]
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    # Use the converse tool-use loop from utils
    answer = utils.llm_converse_with_tools(
        model=client.retrieve_model,
        messages=messages,
        tools=TOOL_SPECS,
        tool_handlers=tool_handlers,
        system=system,
        max_turns=20,
    )

    print(f"\n[answer]: {answer}")
    return answer


if __name__ == "__main__":
    # Verify doc exists
    if not DOC_PATH.exists():
        print(f"ERROR: Document not found at {DOC_PATH}")
        sys.exit(1)

    # Setup
    client = PageIndexClient(workspace=WORKSPACE)

    # Step 1: Load or index document and view tree structure
    print("=" * 60)
    print("Step 1: Load document and view tree structure")
    print("=" * 60)
    doc_name = DOC_PATH.stem
    doc_id = next(
        (
            did
            for did, doc in client.documents.items()
            if doc.get("doc_name") == doc_name
        ),
        None,
    )
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id}")
    else:
        doc_id = client.index(DOC_PATH)
        print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = json.loads(client.get_document_structure(doc_id, depth=1))
    utils.print_tree(structure)

    # Step 2: View document metadata
    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    doc_metadata = client.get_document(doc_id)
    print(f"\n{doc_metadata}")

    # Step 3: Agent Query
    print("\n" + "=" * 60)
    print("Step 3: Agent Query (Bedrock Converse tool-use)")
    print("=" * 60)
    question = "Provide a top-10 list of the biggest funding areas supported by H.R. 7148."
    print(f"\nQuestion: '{question}'")
    query_agent(client, doc_id, question, verbose=True)
