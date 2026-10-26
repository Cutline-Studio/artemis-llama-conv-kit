# llama-conv-kit: Artemis conversational LoRA on Llama 3.1 8B Instruct (prepared, NOT yet run)

Status 2026-10-26 ~1:00 AM ET: **blocked before rental.** There's no Hugging Face token on the box, and
meta-llama/Llama-3.1-8B-Instruct is gated (`gated: manual`, anonymous HTTP 401). Nothing was rented.

To run: put an HF read token (from an account that has accepted the Llama 3.1 license) in `./.hf_token` (chmod 600), then
`bash launch.sh --dry-run` (no spend: checks access + picks the offer), then `bash launch.sh`.

Dataset (`build_conv_sft.py`, seed 567885, deterministic): 900 raw -> 854 after exact dedup -> 781 train / 73 val.
- 349 real examples reused from qwen-kit/sft/artemis_sft.jsonl (doc Q&A extracted from the committed docs, the docs' own
  example dialogues, plus the hand-written identity/governance/tool examples). Qwen identity lines were rewritten for Llama,
  and doc answers gained `[sources: File.md#Section]` citations.
- 505 SYNTHETIC multi-turn dialogues templated from 117 committed doc sections (a567-dense HEAD 56222e8): memory, follow-up,
  clarify-then-answer, preference recall, topic switch, approval flows, and resolving "it" from context. They're labelled `synthetic: true`.
- Val is grouped: Anthropic.md, HTTP_HTTPS.md, Instructions.md and Watch.md are val-only, plus ~8% of the non-doc rows.

Guards: on-instance selfdestruct (HARD = rent + lifetime, or 20 min after DONE), a detached box killer at the lifetime, and a
harvester that pulls out/ (adapter, logs, probes, license files, no base weights) then destroys and verifies by ID. The lifetime
comes from the live $/hr, ~17 GB of download at the host's $/GB, and storage, with a $0.35 margin under $4.00.

License: the Llama 3.1 Community License + Acceptable Use Policy are copied into the adapter dir, with a NOTICE that says "Built with Llama".
If distributed: show "Built with Llama" prominently, and any derived model name must begin with "Llama".
