# Node Description Batch 4 of 4

Graphify is running in assistant/skill mode (no API key). You are the host
assistant (Claude Code / Codex / Gemini CLI). Read the prompt below and write
your JSON answer to the answer file.

## Prompt

You are documenting nodes in a knowledge graph.
For each entry below, write ONE concise factual plain-language sentence
describing what it is or does. Use only the provided context.
For a code symbol (kind=code-symbol — a function, class, or constant),
describe what the function/symbol does based on its name, source location
and neighbors — e.g. "Resolves the configured ontology profile from graphify.yaml.".
For an entity node (any other kind — e.g. a person, place, event, object),
describe what the entity is and its role, grounded in its type, its
relations (neighbors) and the provided citations/evidence — e.g.
"Lady Carfax, a wealthy heiress who disappears en route to Lausanne.".
Ground entity descriptions in the citations/evidence when present; do not
speculate beyond the context, so a node with no supporting context may be
left out of the reply.
Write every description in English (en). Do not switch languages.
No marketing language.
Respond ONLY with a JSON object mapping each node id (as a string) to its
one-sentence description — no prose, no markdown fences.

- "utils_init": "__init__.py" | kind=code-symbol | source=tanitas/backend/utils/__init__.py:L1 | neighbors=[Utility helpers for the tanitas local-G…]
- "utils_init_rationale_1": "Utility helpers for the tanitas local-GPU backend." | kind=entity | source=tanitas/backend/utils/__init__.py:L1 | neighbors=[__init__.py]
- "utils_validators_rationale_1": "backend/utils/validators.py\r ----------------------------\r Input validation help" | kind=entity | source=tanitas/backend/utils/validators.py:L1 | neighbors=[validators.py]
- "utils_validators_rationale_110": "Detect the actual image type from magic bytes and verify it matches\r     the dec" | kind=entity | source=tanitas/backend/utils/validators.py:L110 | neighbors=[_check_magic_bytes()]
- "utils_validators_rationale_144": "Reject files that exceed the configured size limit." | kind=entity | source=tanitas/backend/utils/validators.py:L144 | neighbors=[_check_size()]
- "utils_validators_rationale_158": "Return the lowercase file extension including the leading dot." | kind=entity | source=tanitas/backend/utils/validators.py:L158 | neighbors=[_get_extension()]
- "utils_validators_rationale_58": "Raised when an uploaded image fails validation." | kind=entity | source=tanitas/backend/utils/validators.py:L58 | neighbors=[ValidationError]
- "utils_validators_rationale_65": "Validate an uploaded image by extension, magic bytes, and size.\r \r     Parameter" | kind=entity | source=tanitas/backend/utils/validators.py:L65 | neighbors=[validate_image()]
- "utils_validators_rationale_96": "Reject files whose extension is not in the allowlist." | kind=entity | source=tanitas/backend/utils/validators.py:L96 | neighbors=[_check_extension()]
- "valueerror": "ValueError" | kind=code-symbol | neighbors=[ValidationError]
- "frontend_init": "__init__.py" | kind=code-symbol | source=tanitas/frontend/__init__.py:L1
- "frontend_postcss_config": "postcss.config.js" | kind=code-symbol | source=tanitas/frontend/postcss.config.js:L1
- "frontend_tailwind_config": "tailwind.config.js" | kind=code-symbol | source=tanitas/frontend/tailwind.config.js:L1
- "frontend_vite_config": "vite.config.js" | kind=code-symbol | source=tanitas/frontend/vite.config.js:L1
- "src_main": "main.jsx" | kind=code-symbol | source=tanitas/frontend/src/main.jsx:L1

## Instructions

Write a single JSON object mapping each node id to a one-sentence description
to: c:\Users\icroc\Desktop\texti\POC-MESH\.graphify\description-instructions\batch-003.json

Keep each description factual and concise (one sentence). No markdown, no prose
outside the JSON object. It is acceptable to omit a node if context is
insufficient — but include every node you can ground confidently.

Example answer format:
```json
{
  "node_id_1": "Resolves the configured ontology profile from graphify.yaml.",
  "node_id_2": "Colonel James Barclay, an antagonist in The Crooked Man."
}
```
