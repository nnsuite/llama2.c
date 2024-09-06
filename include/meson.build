llama2c_common_headers = files(
    'api.h',
    'sampler.h',
    'tokenizer.h',
)
llama2c_headers = files('transformer.h')
llama2cq_headers = files('transformer_quantized.h')

install_headers(
    [llama2c_common_headers, llama2c_headers, llama2cq_headers],
    install_dir: llama2c_install_includedir,
)
