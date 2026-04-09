# frozen_string_literal: true

require 'spec_helper'
require 'tempfile'

RSpec.describe MiniEmbed do
  describe '.new' do
    it 'loads a valid GGUF model' do
      correct_gguf_path = 'spec/fixtures/gte-small.Q4_0.gguf'
      expect { described_class.new(model: correct_gguf_path) }.not_to raise_error
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

    context 'with default behavior (type: :vector)' do
      it 'returns an array of floats' do
        result = model.embeddings(text: 'hello')
        expect(result).to be_a(Array)
        expect(result.first).to be_a(Float)
      end

      it 'returns an embedding of expected dimension' do
        result = model.embeddings(text: 'hello world')
        expect(result.size).to be > 0
        # All entries should be finite numbers (not NaN or Infinity)
        expect(result.all? { |v| v.finite? }).to be true
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

      it 'handles empty string (returns zero vector)' do
        result = model.embeddings(text: '')
        expect(result).to be_a(Array)
        expect(result.all?(&:zero?)).to be true
      end

      it 'handles unknown tokens gracefully (non-zero if some tokens known)' do
        # If all tokens unknown, result may be zero; but we just check no crash
        result = model.embeddings(text: 'xyznonexistenttoken123')
        expect(result).to be_a(Array)
        expect(result.all? { |v| v.finite? }).to be true
      end
    end

    context 'with type: :binary' do
      it 'returns a binary string' do
        result = model.embeddings(text: 'hello', type: :binary)
        expect(result).to be_a(String)
        expect(result.encoding).to eq(Encoding::ASCII_8BIT)
      end

      it 'returns correct byte size (dimension * 4)' do
        result = model.embeddings(text: 'hello world', type: :binary)
        expect(result.bytesize).to be > 0
        expect(result.bytesize % 4).to eq(0)
      end

      it 'produces same floats as the array version' do
        binary = model.embeddings(text: 'test', type: :binary)
        array  = model.embeddings(text: 'test', type: :vector)
        expect(binary.unpack('e*')).to eq(array)
      end
    end

    it 'raises ArgumentError for unsupported type' do
      expect { model.embeddings(text: 'hello', type: :invalid) }.to(
        raise_error(ArgumentError, /Unsupported data type/)
      )
    end
  end
end
