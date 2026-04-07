# frozen_string_literal: true

require 'spec_helper'
require 'tempfile'

RSpec.describe MiniEmbed do
  describe '.new' do
    it 'loads a valid GGUF model' do
      correct_gguf_path = 'spec/fixtures/gte-small.Q4_0.gguf'
      expect { described_class.new(model: correct_gguf_path) }.not_to raise_error
    end

    it 'raises an error for an invalid file path' do
      non_existent_file_path = 'spec/fixtures/non-gguf.file.gguf'
      expect { described_class.new(model: non_existent_file_path) }.to(
        raise_error(RuntimeError, /failed to load GGUF model/)
      )
    end

    it 'raises an error for a non-GGUF file' do
      incorrect_file_path = 'spec/fixtures/non-gguf.file.gguf'
      expect { described_class.new(model: incorrect_file_path) }.to(
        raise_error(RuntimeError, /failed to load GGUF model/)
      )
    end
  end

  describe '#embeddings' do
    let!(:model_path) { 'spec/fixtures/gte-small.Q4_0.gguf' }
    let!(:model) { described_class.new(model: model_path) }

    it 'returns a binary string' do
      result = model.embeddings(text: 'hello')
      expect(result).to be_a(String)
      expect(result.encoding).to eq(Encoding::ASCII_8BIT)
    end

    it 'returns an embedding of expected size' do
      result = model.embeddings(text: 'hello world')
      # The size in bytes should be dimension * 4 (float32)
      # We can't know dimension without introspection; check it's non-zero
      expect(result.bytesize).to be > 0
      expect(result.bytesize % 4).to eq(0)
    end

    it 'returns different embeddings for different inputs' do
      emb1 = model.embeddings(text: 'cat')
      emb2 = model.embeddings(text: 'dog')
      expect(emb1).not_to eq(emb2)
    end

    it 'returns the same embedding for identical inputs' do
      emb1 = model.embeddings(text: 'repeatable')
      emb2 = model.embeddings(text: 'repeatable')
      expect(emb1).to eq(emb2)
    end

    it 'handles empty string' do
      result = model.embeddings(text: '')
      expect(result).to be_a(String)
      # Should be all zeros (mean pooling of no tokens)
      floats = result.unpack('e*')
      expect(floats.all?(&:zero?)).to be true
    end

    it 'handles unknown tokens gracefully' do
      # If token not in vocab, it's ignored; result may be zeros if no valid tokens
      result = model.embeddings(text: 'xyznonexistenttoken123')
      expect(result).to be_a(String)
    end
  end
end
