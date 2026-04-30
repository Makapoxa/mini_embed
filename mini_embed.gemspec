# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = 'mini_embed'
  spec.version       = '0.4.1'
  spec.authors       = ['Makapoxa']

  spec.summary       = 'Fast GGUF embedding extraction'
  spec.description   = 'A minimal C extension to load GGUF models and compute token embeddings.'
  spec.homepage      = 'https://github.com/Makapoxa/mini_embed'
  spec.license       = 'MIT'
  spec.required_ruby_version = '>= 3.0.0'

  spec.files         = Dir['lib/**/*.rb', 'ext/**/*.{c,rb}', 'README.md', 'LICENSE.txt']
  spec.extensions    = ['ext/mini_embed/extconf.rb']
  spec.require_paths = ['lib']
  spec.metadata['rubygems_mfa_required'] = 'true'
end
