# frozen_string_literal: true

require 'rake/extensiontask'
require 'rake/clean'

Rake::ExtensionTask.new('mini_embed') do |ext|
  ext.lib_dir = 'lib/mini_embed'
end

task default: :compile

CLEAN.include('lib/mini_embed/*.{so,bundle,dll}')
