# TODO: 'test_all.py' executes the 'run' binary in the current directory.
configure_file(output: 'tokenizer.bin', input: 'tokenizer.bin', copy: true)
executable('testc',
    ['test.c'],
    link_with: libllama2c,
    dependencies: deps
)
