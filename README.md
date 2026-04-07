# mini_embed

Fast, minimal GGUF embedding extractor for Ruby.

## Installation

Add to your Gemfile:

```ruby
gem 'mini_embed'
```
Or install globally:

```sh
gem install mini_embed
```

Usage
```ruby
require 'mini_embed'

model = MiniEmbed.new(model: 'path/to/model.gguf')
embeddings_bin = model.embeddings(text: "hello world")  # => binary ouput
embeddings_array = embeddings_bin.unpack('f*') # => array of float
puts embeddings_array.size                     # => model dimension
```

Supported Quantizations

```
F32, F16

Q4_0, Q4_1

Q5_0, Q5_1

Q8_0, Q8_1

Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
```

## Building the Gem

From the `mini_embed/` directory:

```bash
bundle install
bundle exec rake compile
```


To build the gem file:

```bash
gem build mini_embed.gemspec
```

To install locally:

```bash
gem install ./mini_embed-0.1.0.gem
```
Using in a Rails project
Add to Gemfile:

```ruby
gem 'mini_embed', path: '/path/to/mini_embed'
```

Then `bundle install` and use as above.

## License

MIT License. See [LICENSE](LICENSE).