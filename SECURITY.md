# Security and privacy

This is an unofficial local inference integration, not a public generation service. Bind ComfyUI to loopback unless you separately design authentication, network isolation and applicable safety controls. Do not expose a raw ComfyUI endpoint to the internet.

Only load model and adapter artifacts you trust. The example verification tool checks size/SHA256 without downloading or executing model content; a match does not grant a model license. Keep upstream licenses and safeguards intact.

Raw benchmark histories, logs, compiler traces, prompts and reference media can contain private data. They are ignored by Git for a reason. Public examples must be reviewed and clearly labelled AI-generated. Never attach tokens, `.env` files, credential stores, private references or model weights to issues.

If GitHub private vulnerability reporting is enabled, use it for sensitive reports. Otherwise open a minimal issue requesting a private contact channel **without** exploit details, secrets or personal information. Do not assume a private reporting feature is enabled merely because this policy exists.

CPU tests, including any future CI, do not certify GPU memory safety. Strict GPU backend failures should be reproduced in a fresh process. After an illegal CUDA access, stop that process rather than continuing with a damaged context. Experimental modes remain opt-in and should not be relied on for safety-critical applications.
