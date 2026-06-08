# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Open a [private advisory](https://github.com/agentsynth/agentsynth/security/advisories/new)
on GitHub, or contact a maintainer directly. We aim to respond within a few days.

## Supported versions

AgentSynth is pre-1.0. Security fixes land on `main` and ship in the next release.
Please run a recent version before reporting.

## A note on the code-execution sandbox

`agentsynth.utils.PythonREPL` runs Python to ground `code_execution` trajectories
in real output. It restricts imports to a numeric/data allowlist and blocks a few
obvious escapes, but **it is a convenience, not a security boundary**. Don't feed
it untrusted code on a host you care about. The same applies to pointing a real LLM
judge or generator at untrusted inputs — treat generated tool arguments and code as
untrusted data.
