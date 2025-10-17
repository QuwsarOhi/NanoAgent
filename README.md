# 🧠 NanoAgent — A 135M Parameter Agentic LLM

NanoAgent is a **135M parameter**, **8k context length**, open-source language model designed for **agentic tasks** such as **tool calling**, **instruction following**, and **lightweight reasoning**.  
It’s small enough (~135 MB in 8-bit) to run on **edge devices** like personal laptops, low-memory CPUs, and even wearables — yet smart enough to make tool calls, parse web information, and give structured answers.

---

## 🌍 Real-World Use Cases

- 🕹️ **Runs on edge devices** — laptops, smartwatches, browsers, or CPU-only environments.  
- 🌐 **Parses and answers from the web** — supports tool calling to fetch real-time information.  
- 🔎 **Answers recent questions** with live web search tools.  
- 💬 **Continues conversations** — ideal for assistant or agent frameworks.  
- ⚙️ **Tool calling support** enables chaining multiple tools and parsing results to produce final answers.

---

## ✨ What NanoAgent Supports

| Capability                        | Description                                                                                     | Dataset Source                                                |
|------------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| 💬 Basic conversation              | Hi/hello, casual small talk                                                                     | `HuggingFaceTB/smoltalk`                                       |
| 🌐 Information retrieval           | e.g., *“How to bake a cake?”*, *“Weather in Toronto”* through web search. Extracts answers from information returned by tools (scraping/search)                        | Tool calling + Web Search                                     |
| 🧰 Tool calling                    | Single & multi-tool call with structured explanation                                            | `Locutusque/function-calling-chatml`, `Salesforce/xlam-function-calling-60k` |
| 🧠 Question decomposition          | Breaks complex questions into steps                                                             | `weijie210/gsm8k_decomposed`                                   |
| 🧭 Question classification         | Identifies type of user query (e.g., fact, reasoning, instruction)                              | `microsoft/orca-agentinstruct-1M-v1`                           |
| 📝 Following system prompts       | Responds properly to system-level instructions                                                  | Instruction datasets                                          |
| ✍️ Writing emails and tasks       | Writes emails, structured messages                                                              | `HuggingFaceTB/smoltalk`                                      |

---

## 🧪 Training Overview

- **Base model**: [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)  
- **Fine-tuning method**: [Dynamic Fine-Tuning (DFT)](https://github.com/yongliang-wu/DFT/tree/master)  
- **Platform**: Apple Mac M1 (16 GB) — MLX framework

### 📚 Datasets Used

| Dataset                                                                                  | Purpose                                                                 |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `microsoft/orca-agentinstruct-1M-v1`                                                      | Agentic tasks, classification, instruction following                     |
| `microsoft/orca-math-word-problems-200k`                                                 | Lightweight reasoning, word-level reasoning                              |
| `allenai/tulu-3-sft-personas-instruction-following`                                     | Instruction following with persona                                      |
| `xingyaoww/code-act`                                                                     | ReAct style reasoning and acting                                        |
| `m-a-p/Code-Feedback`                                                                    | Feedback alignment                                                      |
| `HuggingFaceTB/smoltalk`                                                                 | General conversation, system prompt handling                            |
| `HuggingFaceTB/smoltalk/apigen`                                                          | Tool calling stabilization                                             |
| `weijie210/gsm8k_decomposed`                                                             | Question decomposition                                                 |
| `Locutusque/function-calling-chatml`                                                     | Tool call response formatting                                          |
| `Salesforce/xlam-function-calling-60k`                                                   | Stronger function calling coverage                                     |
| `HuggingFaceTB/smoltalk2/SFT/smolagents_toolcalling_traces_think`                         | Web search, scraping, real-time reasoning                               |
| `Jofthomas/hermes-function-calling-thinking-V1`                                          | Tool calling support with thinking |
| `HuggingFaceTB/smoltalk/smol-magpie-ultra` | For python code writing |
---

## 🧭 Key Explorations & Findings

- ✂️ **Dataset deduplication** significantly improved performance by removing noisy or duplicate Q/As.  
 - ✂️ **Shortening the responses** (casual response) and using shorter python code in training improved performance and reduce repeated token generation.
- 🧮 **Word-level reasoning** from `orca-math` enhanced the model’s ability to handle stepwise logic.  
- 🧰 Designing tool calling prompts using **six open-source tool calling datasets** resulted in stronger structured output generation.  
- 🌐 Tool calling integration enabled the model to **extract answers from parsed web data**, supporting up-to-date queries.  

---

## ⚡ Benchmark

| Metric / Task                      | SmolLM2-135M-Instruct | NanoAgent                |
|--------------------------------------|-------------------------|-----------------------------------|
| 🧮 **Parameters**                   | 135M                    | 135M                              |
| 📏 **Context Length**               | 8k                      | 8k                                |
| 📊 **IFEval Score (Overall)**       | ---                    | ---                          |
| 🧰 **Tool Call Tasks**             | ❌ Not Supported        | ✅ Supported                      |
| 🧭 **Instruction Following**       | 🟡 Moderate             | 🟢 Improved                       |
| 🧠 **Reasoning (Light)**          | 🟡 Moderate             | 🟡 Moderate                       |
| 📝 **Training Method**            | Baseline (SFT)          | DFT + Agentic Finetuning         |
| 🧪 **Strength**                   | Instruction following   | Tool call ability + structured outputs |
| ⚠️ **Limitations**               | No tool calling         | Occasional tool errors, still beta |

> *Scores measured using exact match across instruction-following and tool-calling tasks. Tool call accuracy reflects structured format compliance.*

---

## 🧭 Roadmap

- [ ] 📊 Benchmark more agentic tasks  
- [ ] 🧠 Explore GRPO for tool calling improvement  
- [ ] 🔀 Experiment with weight merging  
- [ ] 🧪 Evaluate multi-turn tool chaining  
- [ ] 🧹 Further refine datasets for stability

---

