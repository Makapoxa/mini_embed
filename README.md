# mini_embed

A minimal, dependency‑free C extension for Ruby that loads [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) embedding models and computes text embeddings **locally**.

**⚠️ Important:** This gem is intended for **small projects, prototypes, and hobbyist use**. It allows you to experiment with embeddings without relying on external APIs or cloud costs. **Do not use MiniEmbed in production** – it lacks the performance, scalability, and tokenization robustness of dedicated solutions. For real applications, use a proper inference server like [llama.cpp](https://github.com/ggerganov/llama.cpp) with its HTTP API, or managed services such as OpenAI, Cohere, or Hugging Face.

---

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/Makapoxa/mini_embed/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/Makapoxa/mini_embed/tree/main)

## Why MiniEmbed?

- **Zero external dependencies** – no TensorFlow, PyTorch, or ONNX runtime.
- **Single‑file C extension** – fast loading and mean‑pooled embeddings.
- **Supports all common GGUF quantizations** – from `F32` to `Q2_K`.
- **Works entirely offline** – your data never leaves your machine.
- Perfect for **weekend projects**, **proof‑of‑concepts**, or **learning** about embeddings.

---

## Installation

Add this line to your application's `Gemfile`:

```ruby
gem 'mini_embed'
```

Then execute:

```bash
bundle install
```
Or install it globally:

```bash
gem install mini_embed
```


## Requirements:

A POSIX system (Linux, macOS, BSD) – Windows via WSL2 works.

A C compiler and make (for compiling the native extension).

A GGUF embedding model file (see Where to get models).

## Usage

```ruby
require 'mini_embed'

# Load a GGUF model (F32, F16, Q8_0, Q4_K, etc. are all supported)
model = MiniEmbed.new(model: '/path/to/gte-small.Q8_0.gguf')

# Get embedding as an array of floats (default)
embedding = model.embeddings(text: 'hello world')
puts embedding.size   # e.g. 384
puts embedding[0..4]  # e.g. [0.0123, -0.0456, ...]

# Or get the raw binary string (little‑endian 32‑bit floats)
binary = model.embeddings(text: 'hello world', type: :binary)
embedding_from_binary = binary.unpack('e*')
```

Note: The type parameter is optional – it defaults to :vector which returns a Ruby `Array<Float>`. Use `type: :binary` to get the raw binary string (compatible with the original C extension).


## Simple tokenization note
MiniEmbed uses a naive space‑based tokenizer. This means it splits input on spaces and looks up each token exactly in the model's vocabulary. For models trained with subword tokenization (like BERT), this will not work for out‑of‑vocabulary words.
If you need proper subword tokenization, you can:

- Pre‑tokenize in Ruby using the tokenizers gem and pass token IDs (not yet exposed in the C API, but easy to add).
- Stick to simple vocabulary words that exist in the model (e.g., "text", "hello", "dog").

## Supported Quantization Types

| Type | Description   |
|------|---------------|
| 0    | F32 (float32) |
| 1    | F16 (float16) |
| 2    | Q4_0          |
| 3    | Q4_1          |
| 6    | Q5_0          |
| 7    | Q5_1          |
| 8    | Q8_0          |
| 9    | Q8_1          |
| 10   | Q2_K          |
| 11   | Q3_K          |
| 12   | Q4_K          |
| 13   | Q5_K          |
| 14   | Q6_K          |
| 15   | Q8_K          |

The extension automatically dequantizes the embedding matrix on load, so inference speed is always that of a plain float32 lookup.

Where to get models
Hugging Face offers many GGUF models, e.g.:

- `gte-small`
- `all‑MiniLM‑L6‑v2`

You can convert any safetensors or PyTorch model using the `convert‑hf‑to‑gguf.py` script from llama.cpp.

For testing, we recommend the `gte-small` model (384 dimensions, ~30k vocabulary).

## Limitations (Why this is not production‑ready)

- Single‑threaded, blocking C code – embedding computation runs on the Ruby thread, freezing the interpreter.
- No batching – only one text at a time.
- Space‑based tokenization only – works only for words present exactly in the vocabulary.
- Loads the entire embedding matrix into RAM – for large vocabularies this may consume significant memory.
- No GPU support – CPU only.
- Error handling is minimal – invalid models may crash the Ruby process.

If you need a robust, scalable solution, consider:

- Running llama.cpp as a server (./server -m model.gguf --embeddings) and calling its HTTP endpoint.
- Using a cloud embeddings API (OpenAI, Cohere, VoyageAI, etc.).
- Deploying a dedicated inference service with BentoML or Ray Serve.


## Development & Contributing
Bug reports and pull requests are welcome on GitHub.
To run the tests:

```bash
bundle exec rspec
```

The gem uses rake-compiler to build the extension. After making changes to the C source, run:

```bash
bundle exec rake compile
```

## License
MIT License. See [LICENSE](LICENSE).
