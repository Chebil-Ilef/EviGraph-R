import argparse
import asyncio
import os
from importlib.metadata import version


def _cmd_query(args):
    if args.qdrant_url:
        os.environ["QDRANT_URL"] = args.qdrant_url
    from evigraph.api.runner import WorkflowRunner
    from evigraph.api.schemas import PipelineConfig, QueryRequest
    runner = WorkflowRunner(model_key=args.model)
    request = QueryRequest(
        query=args.question,
        config=PipelineConfig(top_k=args.top_k),
    )
    result = asyncio.run(runner.run_query(request))
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.answer)


def _cmd_serve(args):
    if args.qdrant_url:
        os.environ["QDRANT_URL"] = args.qdrant_url
    import uvicorn
    uvicorn.run(
        "evigraph.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )


def main():
    parser = argparse.ArgumentParser(prog="evigraph")
    parser.add_argument("--version", action="version", version=version("evigraph-r"))
    sub = parser.add_subparsers(dest="command", required=True)

    qp = sub.add_parser("query", help="Run a one-shot query against the evidence graph")
    qp.add_argument("question")
    qp.add_argument("--model", default="bge-m3")
    qp.add_argument("--top-k", type=int, default=15)
    qp.add_argument("--qdrant-url", default=None, help="Override QDRANT_URL")
    qp.add_argument("--json", action="store_true", help="Print full JSON response")
    qp.set_defaults(func=_cmd_query)

    sp = sub.add_parser("serve", help="Start the FastAPI server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.add_argument("--qdrant-url", default=None, help="Override QDRANT_URL")
    sp.set_defaults(func=_cmd_serve)

    args = parser.parse_args()
    args.func(args)
