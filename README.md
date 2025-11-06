# 🧠 NanoAgent — A 135M Parameter Agentic LLM

NanoAgent is a **135M parameter**, **8k context length**, open-source language model designed for **agentic tasks** such as **tool calling**, **instruction following**, and **lightweight reasoning**.  
It’s small enough (~135 MB in 8-bit) to run on **edge devices** like personal laptops, low-memory CPUs, and even wearables — yet smart enough to make tool calls, parse web information, and give structured answers.

Quick inference resource: [here](notebooks/inference.ipynb)

## 🌍 Real-World Use Cases

- 🕹️ **Runs on edge devices** — laptops, smartwatches, browsers, or CPU-only environments.  
- 🌐 **Parses and answers from the web** — supports tool calling to fetch real-time information.  
- 🔎 **Answers recent questions** with live web search tools.  
- 💬 **Continues conversations** — ideal for assistant or agent frameworks.  
- ⚙️ **Tool calling support** enables chaining multiple tools and parsing results to produce final answers.


## ✨ What NanoAgent Supports

| Capability                        | Description                                                                                     | 
|------------------------------------|--------------------------------------------------------------------------------------------------|
| 💬 Basic conversation              | Casual small talk                                                                     |
| 🌐 Information retrieval           | e.g., *“How to bake a cake?”*, *“Weather in Toronto”* through web search. Extracts answers from information returned by tools (scraping/search)                        |
| 🧰 Tool calling                    | Single & multi-tool call with structured explanation                                            |
| 🧠 Question decomposition          | Breaks complex questions into steps                                                             | 
| 🧭 Question classification         | Identifies type of user query (e.g., fact, reasoning, instruction)                              |
| 📝 Following system prompts       | Responds properly to system-level instructions                                                  | 
| ✍️ Writing emails and tasks       | Writes emails, structured messages                                                              | 
---

## 🧪 Training Overview

- **Base model**: [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)  
- **Fine-tuning method**: ~~[Dynamic Fine-Tuning (DFT)](https://github.com/yongliang-wu/DFT/tree/master)~~ Supervised Fine-Tuning
- **Platform**: Apple Mac M1 (16 GB) — MLX framework

### 📚 Datasets Used

This model was trained using a combination of datasets under different open licenses.  
Each dataset retains its original license, and use of those datasets is subject to their respective terms.

| Dataset                                                                                  | Purpose                                                                 | License |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|----------------|
| [microsoft/orca-agentinstruct-1M-v1](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)                                                      | RAG, MCQ answering, JSON parsing, Text classification, instruction following                     | Community Data License Agreement – Permissive, Version 2.0 |
| [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)                                                 | Lightweight reasoning, word-level reasoning | MIT                              |
| [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)                                     | Instruction following with persona                                      | Open Data Commons License Attribution family |
| [xingyaoww/code-act](https://huggingface.co/datasets/xingyaoww/code-act)                                                                     | ReAct style reasoning and acting                                        | Apache-2.0 |
| [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)                                                                    | Feedback alignment                                                      | Apache-2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)                                                                 | General conversation, system prompt handling                            | Apache-2.0 |
| [HuggingFaceTB/smoltalk/apigen](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)                                                          | Tool calling stabilization                                             | Creative Commons Attribution 4.0 (was sourced from [1](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), [2](https://huggingface.co/datasets/argilla/apigen-function-calling)) |
| [weijie210/gsm8k_decomposed](https://huggingface.co/datasets/weijie210/gsm8k_decomposed)                                                             | Question decomposition                                                 | - |
| [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml)                                                     | Tool call response formatting                                          | Apache-2.0 |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)  | Stronger function calling coverage                                     | Creative Commons Attribution 4.0 |
| [HuggingFaceTB/smoltalk2/SFT/smolagents_toolcalling_traces_think](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/viewer/SFT/smolagents_toolcalling_traces_think)                         | Web search, scraping, real-time reasoning                               | Apache-2.0 |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)                                          | Tool calling support with thinking | Apache-2.0 |
| [HuggingFaceTB/smoltalk/smol-magpie-ultra](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/smol-magpie-ultra) | For python code writing | Apache-2.0 |


## 🧭 Key Explorations & Findings

- ✂️ **Dataset deduplication** significantly improved performance by removing noisy or duplicate Q/As.  
 - ✂️ **Shortening the responses** (casual response) and using shorter python code in training improved performance and reduce repeated token generation.
- 🧮 **Word-level reasoning** from `orca-math` enhanced the model’s ability to handle stepwise logic.  
- 🧰 Designing tool calling prompts using **six open-source tool calling datasets** resulted in stronger structured output generation.  
- 🌐 Tool calling integration enabled the model to **extract answers from parsed web data**, supporting up-to-date queries.  


## ⚡ Benchmark

| Metric / Task                      | SmolLM2-135M-Instruct | NanoAgent                |
|--------------------------------------|-------------------------|-----------------------------------|
| 🧮 **Parameters**                   | 135M                    | 135M                              |
| 📏 **Context Length**               | 8k                      | 8k                                |
| 📊 **IFEval Score (Overall)**       | ---                    | ---                          |
| 🧰 **Tool Call Tasks**             | ❌ Not Supported        | ✅ Supported                      |
| 🧭 **Instruction Following**       | 🟡 Moderate             | 🟢 Improved                       |
| 🧠 **Reasoning (Light)**          | 🟡 Moderate             | 🟡 Moderate                       |
| 📝 **Training Method**            | Baseline (SFT)          | SFT + Agentic Finetuning         |
| 🧪 **Strength**                   | Instruction following   | Tool call ability + structured outputs |
| ⚠️ **Limitations**               | No tool calling         | Occasional tool errors, still beta |


## 🧭 Roadmap

- [ ] 📊 Benchmark more agentic tasks  
- [ ] 🧠 Explore GRPO for tool calling improvement  
- [ ] 🔀 Experiment with weight merging  
- [ ] 🧪 Evaluate multi-turn tool chaining  
- [ ] 🧹 Further refine datasets for stability

---

## 📄 License

This project (code, model weights, and training recipes) is licensed under the [Apache License 2.0](./LICENSE).

## 📢 Notice

- Model & code are © [quwsarohi](https://github.com/QuwsarOhi), licensed under Apache 2.0.  
- Portions of the training data were sourced from third-party datasets under CDLA-P 2.0, MIT, CC-BY 4.0, ODC-BY, and Apache 2.0.  
- The licensors of these datasets do **not endorse** this project or its outputs.  
- If you redistribute or fine-tune this model, ensure your use complies with all applicable dataset licenses.


