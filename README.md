# Nagrik-Sahayak (Citizen Helper)
**AI for Bharat Hackathon - Idea Submission**

> *"Bridging the gap between Governance and the Grassroots through Voice AI."*

![Project Status](https://img.shields.io/badge/Status-Idea%20Phase-blue) ![Track](https://img.shields.io/badge/Track-AI%20for%20Communities-green) ![Tech](https://img.shields.io/badge/AI-GenAI%20%7C%20RAG-orange)

## Overview
**Nagrik-Sahayak** is a proposed AI-powered, voice-first mobile application designed to empower rural Indian citizens. It acts as a personal assistant that listens to users in their native dialects, searches through complex government scheme documents, and explains benefits and eligibility in simple spoken language.

### The Problem
* **Language Barrier:** Most government schemes are in English or formal Hindi, while rural users speak local dialects.
* **Literacy:** Many target beneficiaries cannot read complex text or fill out long forms.
* **Complexity:** Finding the right scheme among thousands of PDFs is overwhelming.

### Our Solution
An intelligent "Voice-to-Voice" bot that uses:
1.  **Bhashini API** for accurate Indian language translation and transcription.
2.  **RAG (Retrieval Augmented Generation)** to fetch facts *only* from official government PDFs (zero hallucinations).
3.  **Generative AI** to hold a natural conversation and check eligibility.

## Architecture & Design
> **Note:** Detailed technical specifications can be found in the `design.md` and `requirements.md` files in this repository.

### Tech Stack Strategy
* **Frontend:** React PWA (Mobile First, Offline Ready)
* **Voice/Translation:** **Bhashini API** (Govt of India Stack) + OpenAI Whisper
* **LLM Orchestration:** LangChain
* **Reasoning Engine:** Google Gemini 1.5 Pro
* **Knowledge Base:** Pinecone (Vector Database for Scheme PDFs)

### High-Level Data Flow
1.  **User Speaks** (e.g., in Bhojpuri) -> **Transcribed & Translated** to English.
2.  **AI Search:** Vector DB retrieves relevant scheme rules (e.g., *PM Kisan Yojana*).
3.  **Reasoning:** LLM checks if the user meets the criteria (e.g., land size < 2 hectares).
4.  **Response:** Answer is generated -> **Translated back** to Bhojpuri -> **Played as Audio**.

## Repository Structure
This repository contains the core design artifacts for the hackathon submission:

* [`requirements.md`](./requirements.md): Detailed user stories, functional requirements, and success metrics.
* [`design.md`](./design.md): System architecture, API endpoints, and database schema.

## Key Features (Proposed)
* **Vernacular Voice Support:** "Speak in your language, hear in your language."
* **Strict RAG Guardrails:** Answers are grounded in official documents to prevent misinformation.
* **Offline Mode:** Critical for rural areas with spotty internet connectivity.
* **Application Assistant:** Step-by-step voice guidance to fill out forms.

## Future Roadmap
* **Phase 1 (Hackathon):** Prototype with Voice Input + RAG Search for 5 major schemes.
* **Phase 2:** Integration with WhatsApp Business API.
* **Phase 3:** Direct "Auto-Fill" integration with government portals.

### Team
* **Team Lead:** Archa Vivek
* **Track:** AI for Communities, Access & Public Impact

*Submitted for the AI for Bharat Hackathon 2026.*
