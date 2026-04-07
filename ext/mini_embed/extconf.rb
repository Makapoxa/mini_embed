# frozen_string_literal: true

require 'mkmf'

# Ensure we have necessary headers and functions
have_header('sys/mman.h')
have_header('stdint.h')
have_func('mmap')
have_func('munmap')

# Create the Makefile
create_makefile('mini_embed/mini_embed')
