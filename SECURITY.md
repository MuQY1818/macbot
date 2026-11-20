# Security policy

[简体中文](SECURITY_zh.md)

Report vulnerabilities through [GitHub private vulnerability reporting][report]. From the repository's
**Security → Advisories** tab, choose **Report a vulnerability**. Include the affected version,
impact and steps to reproduce, using fake mode or simulation where possible.
Do not post vulnerability details or credentials in a public issue. If private reporting is
unavailable, open a public issue asking the maintainer `@MuQY1818` to enable it, without
disclosing the vulnerability.

macbot can move a physical arm. Validate changes in `--fake` or the simulator first. For authorized
hardware work, clear the workspace, keep the power cutoff within reach, and begin at low speed.
If motion becomes unsafe, stop the arm before collecting logs or filing a report.

[report]: https://github.com/MuQY1818/macbot/security/advisories/new
