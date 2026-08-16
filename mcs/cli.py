"""Console entrypoint for Agent Compat Lab."""

import argparse
import dataclasses
import os
import sys

_CAPABILITIES = [
    "core",
    "special_tokens",
    "streaming",
    "tools",
    "multimodal",
    "multiturn",
    "reasoning",
    "robustness",
    "perf",
]
_PROFILES = ["generic", "codex", "hermes", "openclaw", "model"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-compat",
        description="Test whether an OpenAI-style endpoint works with coding agents.",
    )
    parser.add_argument(
        "--profile",
        choices=_PROFILES,
        default="generic",
        help="client path to test (default: generic Chat Completions)",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="target model id (repeat only with --profile model)",
    )
    parser.add_argument("--base-url", help="API root including /v1 when required")
    parser.add_argument(
        "--allow-no-auth",
        action="store_true",
        help="allow an empty API key for local Ollama or mock endpoints",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument(
        "--markdown", metavar="PATH", help="also write an agent-profile Markdown report"
    )
    parser.add_argument(
        "--record-responses",
        metavar="DIR",
        dest="record_dir",
        help="record redacted request and response bodies for each check",
    )

    model = parser.add_argument_group("full model profile")
    model.add_argument(
        "--capability",
        "--suite",
        dest="capability",
        help="with --profile model, run only this capability (comma-separated ok)",
    )
    model.add_argument("--junit", metavar="PATH", help="write full-suite JUnit XML")
    model.add_argument(
        "--detail",
        action="store_true",
        help="list failure reasons in a full-suite multi-model comparison",
    )
    model.add_argument(
        "--spec",
        type=str.lower,
        choices=["openai", "dev"],
        default="openai",
        help="full-suite API surface: openai (default) or dev extensions",
    )
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(argv)

    models = args.model or []
    if args.profile != "model" and len(models) > 1:
        parser.error("repeated --model is supported only with --profile model")
    if len(models) == 1:
        os.environ["ACL_MODEL"] = models[0]
        os.environ["MCS_MODEL"] = models[0]
    if args.base_url:
        os.environ["ACL_API_BASE"] = args.base_url
        os.environ["MCS_API_BASE"] = args.base_url
    if args.allow_no_auth:
        os.environ["ACL_ALLOW_NO_AUTH"] = "1"
    if args.record_dir:
        if os.path.exists(args.record_dir):
            parser.error(
                f"--record-responses directory already exists: {args.record_dir} "
                f"(refusing to overwrite; choose a new path or remove it)"
            )
        record_dir = os.path.abspath(args.record_dir)
        os.environ["ACL_RECORD_DIR"] = record_dir
        os.environ["MCS_RECORD_DIR"] = record_dir

    from .config import Config

    config = Config.from_env()
    if not config.api_base:
        parser.error("no API base URL (pass --base-url or set ACL_API_BASE)")
    if not config.model:
        parser.error("no model id (pass --model or set ACL_MODEL)")
    if not config.api_key and not args.allow_no_auth:
        parser.error(
            "no API key (set ACL_API_KEY, or use --allow-no-auth for a local endpoint)"
        )

    if args.profile != "model":
        if args.capability or args.junit or args.detail or args.spec != "openai":
            parser.error(
                "--capability/--junit/--detail/--spec apply only to --profile model"
            )
        from .agent_checks import report_agent

        return report_agent(
            config,
            args.profile,
            as_json=args.json,
            markdown_path=args.markdown,
        )

    if args.markdown:
        parser.error("--markdown is currently available for agent profiles only")
    capability = None
    if args.capability:
        wanted = [item.strip() for item in args.capability.split(",") if item.strip()]
        unknown = [item for item in wanted if item not in _CAPABILITIES]
        if unknown:
            parser.error(
                f"unknown capability: {', '.join(unknown)} "
                f"(choose from: {', '.join(_CAPABILITIES)})"
            )
        capability = " or ".join(wanted)

    from .capabilities import report, report_compare

    if len(models) > 1:
        configs = [dataclasses.replace(config, model=model) for model in models]
        return report_compare(
            configs,
            capability=capability,
            spec=args.spec,
            as_json=args.json,
            detail=args.detail,
        )
    return report(
        config,
        capability=capability,
        spec=args.spec,
        as_json=args.json,
        junit=args.junit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
