# frozen_string_literal: true

require 'mini_embed/mini_embed'

class MiniEmbed
  # @param text [String] - text to extract embeddings from
  # @param type [Symbol, nil] - :binary or :vector - type of data you want to receive
  # @return [String, <Float>] - type == :binary - binary string, type == :vector - array of floats
  def embeddings(text:, type: :vector)
    binary_data = embed(text: text) # call original C method

    return binary_data if type == :binary
    return binary_data.unpack('e*') if type == :vector

    raise ArgumentError, "Unsupported data type: #{type}"
  end
end
