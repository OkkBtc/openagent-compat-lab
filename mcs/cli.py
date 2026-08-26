"""Console entrypoint for openagent-compat-lab."""

import argparse
import dataclasses
import json
import math
import os
import sys
from importlib.metadata import version

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
_AGENT_PROFILE_PATHS = {
    "generic": "Chat Completions",
    "codex": "Responses API",
    "hermes": "Chat Completions",
    "openclaw": "Chat Completions stream",
}
_PROFILES = [*_AGENT_PROFILE_PATHS, "all", "model"]


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
        "--version",
        action="version",
        version=f"%(prog)s {version('openagent-compat-lab')}",
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
        "--list-profiles",
        action="store_true",
        help="list supported agent profiles and API paths without credentials",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="list checks for the selected agent profiles without sending requests",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="show the redacted effective agent configuration without sending requests",
    )
    parser.add_argument(
        "--check",
        action="append",
        metavar="NAME",
        help="run only this check for one named agent profile; repeat to select more",
    )
    parser.add_argument(
        "--skip-check",
        action="append",
        metavar="NAME",
        help="skip this check for one named agent profile; repeat to skip more",
    )
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
        "--json-output",
        metavar="PATH",
        help="write an agent-profile JSON report without changing stdout format",
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

    if args.list_profiles:
        if any(argument not in {"--list-profiles", "--json"} for argument in argv):
            parser.error("--list-profiles cannot be combined with other options")
        from .agent_checks import MATRIX_PROFILES, agent_check_names

        payload = {
            "profiles": {
                profile: {
                    "api_path": api_path,
                    "checks": len(agent_check_names(profile)),
                    "included_in_all": profile in MATRIX_PROFILES,
                }
                for profile, api_path in _AGENT_PROFILE_PATHS.items()
            },
            "matrix_profile": {"name": "all", "profiles": list(MATRIX_PROFILES)},
            "full_suite_profile": "model",
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("Supported agent profiles")
            for profile, details in payload["profiles"].items():
                matrix = "; included in all" if details["included_in_all"] else ""
                print(
                    f"  {profile}: {details['api_path']} "
                    f"({details['checks']} checks{matrix})"
                )
            print(f"  all: {', '.join(payload['matrix_profile']['profiles'])}")
            print("  model: inherited full model compatibility suite")
        return 0

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

    if profile == "model":
        if args.list_checks:
            parser.error("--list-checks applies only to agent profiles")
        if args.show_config:
            parser.error("--show-config applies only to agent profiles")
        if args.check:
            parser.error("--check applies only to one named agent profile")
        if args.skip_check:
            parser.error("--skip-check applies only to one named agent profile")
        if args.json_output:
            parser.error("--json-output applies only to agent profiles")
    else:
        if args.capability or args.detail or args.spec != "openai":
            parser.error("--capability/--detail/--spec apply only to --profile model")
        from .agent_checks import MATRIX_PROFILES, agent_check_names

        selected_profiles = list(MATRIX_PROFILES) if profile == "all" else profiles
        checks_by_profile = {
            selected: agent_check_names(selected) for selected in selected_profiles
        }
        if args.check and args.skip_check:
            parser.error("--check and --skip-check cannot be combined")
        if args.check:
            if profile == "all" or len(profiles) != 1:
                parser.error("--check requires exactly one named agent profile")
            duplicates = [name for name in args.check if args.check.count(name) > 1]
            if duplicates:
                parser.error(f"duplicate --check: {duplicates[0]}")
            unknown = [
                name for name in args.check if name not in checks_by_profile[profile]
            ]
            if unknown:
                parser.error(f"unknown check for {profile}: {unknown[0]}")
        run_check_names = args.check
        if args.skip_check:
            if profile == "all" or len(profiles) != 1:
                parser.error("--skip-check requires exactly one named agent profile")
            duplicates = [
                name for name in args.skip_check if args.skip_check.count(name) > 1
            ]
            if duplicates:
                parser.error(f"duplicate --skip-check: {duplicates[0]}")
            unknown = [
                name
                for name in args.skip_check
                if name not in checks_by_profile[profile]
            ]
            if unknown:
                parser.error(f"unknown check for {profile}: {unknown[0]}")
            skipped = set(args.skip_check)
            run_check_names = [
                name for name in checks_by_profile[profile] if name not in skipped
            ]
            if not run_check_names:
                parser.error("--skip-check cannot skip every check")
        if args.list_checks:
            if any(
                [
                    args.check,
                    args.skip_check,
                    args.show_config,
                    args.fail_fast,
                    args.markdown,
                    args.junit,
                    args.json_output,
                    args.record_dir,
                ]
            ):
                parser.error(
                    "--list-checks cannot be combined with run or report options"
                )
            if args.json:
                print(
                    json.dumps(
                        {"profiles": checks_by_profile},
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                for selected, names in checks_by_profile.items():
                    print(f"{selected}:")
                    for name in names:
                        print(f"  {name}")
            return 0
        if args.show_config and any(
            [
                args.fail_fast,
                args.markdown,
                args.junit,
                args.json_output,
                args.record_dir,
            ]
        ):
            parser.error("--show-config cannot be combined with run or report options")

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

    if args.show_config:
        from .redaction import redact

        planned_checks = checks_by_profile
        if run_check_names is not None:
            planned_checks = {profile: tuple(run_check_names)}
        payload = {
            "profile": ("selected" if len(profiles) > 1 else profile),
            "profiles": planned_checks,
            "model": redact(config.model, config.api_key),
            "api_base": redact(config.api_base, config.api_key),
            "timeout_seconds": config.timeout,
            "auth": "configured" if config.api_key else "disabled",
            "requests_sent": False,
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("Resolved agent compatibility plan")
            print(f"  model:    {payload['model']}")
            print(f"  endpoint: {payload['api_base']}")
            print(f"  timeout:  {payload['timeout_seconds']} seconds")
            print(f"  auth:     {payload['auth']}")
            print("  checks:")
            for selected, names in planned_checks.items():
                print(f"    {selected}: {', '.join(names)}")
            print("  requests: 0 (configuration only)")
        return 0

    if profile != "model":
        from .agent_checks import report_agent, report_agent_matrix

        report_options = {
            "as_json": args.json,
            "markdown_path": args.markdown,
            "junit_path": args.junit,
            "json_path": args.json_output,
            "fail_fast": args.fail_fast,
        }
        if profile == "all":
            return report_agent_matrix(config, **report_options)
        if len(profiles) > 1:
            return report_agent_matrix(config, profiles, **report_options)
        return report_agent(
            config, profile, check_names=run_check_names, **report_options
        )

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
