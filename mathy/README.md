```
ollama run qwen3:0.6b
ollama run gemma3:1b
ollama run gemma3:4b
ollama serve
```

gemma3:1b is a non thinking model and quickly produces the incorrect answer, so this is good.

Results:
On 50 equations in `equations.txt` the question asked with a simple prompt, the model Gemma 4b scores 10% accuracy. In agent mode it scores 92% accuracy.

This small 3.3GiB model is MUCH better at solving math equations in agent mode than in simple Q/A mode.