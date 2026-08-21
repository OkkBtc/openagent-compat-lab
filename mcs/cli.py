"""Console entrypoint for openagent-compat-lab."""

import argparse
import dataclasses
import math
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
_PROFILES = ["generic", "codex", "hermes", "openclaw", "all", "model"]


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-compat",
        description="Probe selected OpenAI-style protocol paths used by coding agents.",
    )
    parser.add_argument(
        "--profile",
        choices=_PROFILES,
        action="append",
        help=(
            "client path to test; repeat named profiles for a selected matrix "
            "(default: generic Chat Completions)"
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        help="target model id (repeat only with --profile model)",
    )
    parser.add_argument("--base-url", help="API root including /v1 when required")
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        metavar="SECONDS",
        help="per-request timeout in seconds (default: ACL_TIMEOUT or 60)",
    )
    parser.add_argument(
        "--allow-no-auth",
        action="store_true",
        help="allow an empty API key for local Ollama or mock endpoints",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop agent-profile probes after the first failed or broken check",
    )
    parser.add_argument(
        "--markdown", metavar="PATH", help="also write an agent-profile Markdown report"
    )
    parser.add_argument("--junit", metavar="PATH", help="write a JUnit XML report")
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

    profiles = args.profile or ["generic"]
    duplicates = [profile for profile in profiles if profiles.count(profile) > 1]
    if duplicates:
        parser.error(f"duplicate --profile: {duplicates[0]}")
    if len(profiles) > 1 and any(profile in {"all", "model"} for profile in profiles):
        parser.error(
            "--profile all and --profile model cannot be combined with other profiles"
        )
    profile = profiles[0]

    models = args.model or []
    if profile != "model" and len(models) > 1:
        parser.error("repeated --model is supported only with --profile model")
    if len(models) == 1:
        os.environ["ACL_MODEL"] = models[0]
        os.environ["MCS_MODEL"] = models[0]
    if args.base_url:
        os.environ["ACL_API_BASE"] = args.base_url
        os.environ["MCS_API_BASE"] = args.base_url
    if args.timeout is not None:
        os.environ["ACL_TIMEOUT"] = str(args.timeout)
        os.environ["MCS_TIMEOUT"] = str(args.timeout)
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

    if profile != "model":
        if args.capability or args.detail or args.spec != "openai":
            parser.error("--capability/--detail/--spec apply only to --profile model")
        from .agent_checks import report_agent, report_agent_matrix

        report_options = {
            "as_json": args.json,
            "markdown_path": args.markdown,
            "junit_path": args.junit,
            "fail_fast": args.fail_fast,
        }
        if profile == "all":
            return report_agent_matrix(config, **report_options)
        if len(profiles) > 1:
            return report_agent_matrix(config, profiles, **report_options)
        return report_agent(config, profile, **report_options)

    if args.fail_fast:
        parser.error("--fail-fast is currently available for agent profiles only")
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
