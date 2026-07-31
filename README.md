AdCust - Adapter Customizer

AdCust is a desktop application for training and testing lightweight LLM adapters. It is designed for users who want to customize the behavior, tone, knowledge, or response style of an open model without full fine-tuning the base model.

The project focuses on a practical workflow:

```text
Local dataset -> Remote GPU training -> LoRA / QLoRA adapter -> Local inference testing
```

## Project Status

AdCust is currently in active development.

- `v1.0.0`: first portable Windows release package created through GitHub Releases.
- Current development branch: `feature/adcust-training-workbench-v2`.
- Current focus: training workflow UX, adapter version management, dataset validation, remote workspace monitoring, and better training observability.

This repository is intentionally structured as a monorepo so the desktop UI, backend API, and reusable training logic can evolve independently while still being tested together.

## Why This Project Exists

Fine-tuning even small LLMs locally can be unrealistic on consumer hardware. AdCust solves that by keeping the user experience local while moving GPU-heavy adapter training to a user-managed remote Linux machine.

AdCust is provider-agnostic by design. It does not hardcode RunPod, AWS, Vast.ai, Lambda, or any other cloud provider. The only assumption is:

> A reachable Linux machine with an NVIDIA GPU and SSH access.

That machine can come from RunPod, Vast.ai, AWS, a local server, a university machine, or any future GPU provider.

## Key Features

- Tauri desktop application with a custom professional UI.
- FastAPI backend with clear API boundaries.
- Reusable `adcust_logic` package for core domain logic.
- Remote SSH-based training workflow.
- Hugging Face Transformers + PEFT LoRA / QLoRA training path.
- Strict local/remote base model identity binding through model manifests.
- Adapter signature metadata for traceability.
- Local inference testing with adapter mount/unmount support.
- Conversation history support during inference.
- Training progress, logs, loss metrics, and remote status feedback.
- Compute node registry for user-managed SSH GPU machines.
- GitHub Actions CI for backend build, unit tests, and integration tests.

## Current V2 Workbench Development

The current feature branch expands AdCust into a more complete training workbench:

- Training target selection: identity binding, style modeling, preference alignment, reasoning pattern, or knowledge injection.
- Required run notes for adapter version history.
- Adapter version registry with local bookmark files.
- Dataset format validation for externally generated training data.
- Document workflow support for JSON, JSONL, TXT, Markdown, and PDF inputs.
- Remote pod workspace inspection and cleanup through SSH.
- Loss target and stable-step early stopping.
- Token pump for smoother local inference streaming.

## Architecture

```text
apps/
  adcust-desktop/    Tauri + React desktop client
  adcust-backend/    FastAPI backend API
  adcust-launcher/   Local launcher utilities

packages/
  adcust-logic/      Reusable domain logic, models, schemas, services, remote training scripts

docs/
  architecture/      Architecture notes and diagrams
  release/           Release process documentation

.github/workflows/
  backend-ci.yml     Backend build, unit test, and integration test pipeline
```

## Training Workflow

1. The user creates or starts a GPU server through a provider website.
2. The user registers the SSH connection in AdCust as a compute node.
3. AdCust verifies SSH connectivity.
4. AdCust checks and prepares the remote Python training environment.
5. AdCust uploads the dataset, config, manifest, and training script.
6. The remote machine trains a LoRA / QLoRA adapter.
7. AdCust streams logs, progress, and loss metrics.
8. AdCust downloads the generated adapter.
9. The user manually stops or destroys the cloud server on the provider website.

AdCust does not pretend to control provider billing or pod lifecycle. It only manages SSH-based training operations.

## Technology Stack

- Desktop: Tauri, React, TypeScript, Vite
- Backend: Python, FastAPI, Pydantic
- Training: Hugging Face Transformers, PEFT, LoRA / QLoRA
- Remote execution: SSH
- CI: GitHub Actions
- Testing: pytest, unit tests, integration tests, fake remote server test infrastructure

## CI Pipeline

The backend CI pipeline has three stages:

1. Build: compile backend/package sources and build the reusable `adcust_logic` package.
2. Unit Tests: run focused backend and domain logic tests.
3. Integration Tests: run backend API integration tests with fake infrastructure.

This keeps the CI lightweight and avoids downloading GPU libraries or model weights during normal pull request validation.

## Release

The first GitHub Release is `AdCust v1.0.0`.

It provides a Windows portable package intended for simple evaluation:

```text
Download ZIP -> Extract locally -> Run AdCust
```

The release package is separate from active development work. New features are developed on feature branches first, then merged through the normal branch flow before a new release is created.

## Engineering Highlights

This project demonstrates:

- Desktop application architecture with a local backend.
- Provider-agnostic cloud GPU workflow design.
- Separation between frontend, backend, and reusable package logic.
- Domain-oriented backend design with rich models, DTO schemas, services, and wrapped exceptions.
- Remote execution and file synchronization over SSH.
- Practical CI design for a GPU-related project without requiring GPU runners.
- Release packaging discipline for a Windows desktop application.

## Intended Audience

AdCust is built for experimentation with adapter-based LLM customization, especially when the user has limited local GPU resources but can access temporary remote GPU compute.

It is also a portfolio project demonstrating full-stack product engineering, ML infrastructure awareness, desktop application packaging, and production-minded workflow design.
