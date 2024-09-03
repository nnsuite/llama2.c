project('llama2.c', 'c',
  version : '0.2024.05.30',
  license: 'MIT',
  license_files: 'LICENSE',
  default_options : [
    'werror=true',
    'warning_level=3',
  ]
)

cc = meson.get_compiler('c')

omp_dep = dependency('openmp', required: false)
libm_dep = cc.find_library('m')

deps = [libm_dep]

# TODO: Build for Windows is not supported yet.
if target_machine.system() == 'windows'
  src_run += files('win.c')
endif

if not get_option('openmp-support').disabled()
  if omp_dep.found()
    deps += omp_dep
    add_project_arguments('-D_SUPPORT_OPENMP_', language: 'c')
  else
    warning('openmp-support is enabled, but failed to find openmp.')
  endif
endif

src_core = files('tokenizer.c', 'sampler.c', 'util.c', 'api.c')
src_llama2 = src_core + files('transformer.c')
src_llama2_quantized = src_core + files('transformer_quantized.c')

libllama2c = shared_library('llama2c',
    src_llama2,
    dependencies: deps)
libllama2cq = shared_library('llama2cq',
    src_llama2_quantized,
    dependencies: deps)

opt_gen_bin = get_option('generate-executables')
if opt_gen_bin.contains('all')
  opt_gen_bin += ['fp32', 'quantized']
endif

if opt_gen_bin.contains('fp32')
  executable('run',
      ['main.c'],
      link_with: libllama2c,
      dependencies: deps)
endif

if opt_gen_bin.contains('quantized')
  executable('runq',
      ['main.c'],
      link_with: libllama2cq,
      dependencies: deps)
endif

if get_option('enable-test')
  subdir('test')
endif
