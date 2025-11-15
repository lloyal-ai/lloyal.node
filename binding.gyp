{
  "targets": [
    {
      "target_name": "lloyal",
      "sources": [
        "src/binding.cpp",
        "src/BackendManager.cpp",
        "src/SessionContext.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(node -p \"require('fs').existsSync('vendor/liblloyal') ? 'vendor/liblloyal/include' : 'liblloyal/include'\")",
        "<!@(node -p \"require('fs').existsSync('vendor/liblloyal') ? 'vendor/liblloyal/include/lloyal' : 'liblloyal/include/lloyal'\")",
        "include"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": ["-fno-exceptions", "-fno-rtti"],
      "cflags_cc!": ["-fno-exceptions", "-fno-rtti"],
      "conditions": [
        [
          "OS=='mac'",
          {
            "xcode_settings": {
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "GCC_ENABLE_CPP_RTTI": "YES",
              "CLANG_CXX_LIBRARY": "libc++",
              "MACOSX_DEPLOYMENT_TARGET": "10.15",
              "OTHER_CPLUSPLUSFLAGS": [
                "-std=c++20",
                "-stdlib=libc++"
              ],
              "OTHER_LDFLAGS": [
                "-Wl,-rpath,<!@(node -p \"require('fs').existsSync('vendor/llama.cpp') ? '<(module_root_dir)/vendor/llama.cpp/build-apple' : '<(module_root_dir)/llama.cpp/build-apple'\")"
              ]
            },
            "libraries": [
              "<!@(node -p \"require('fs').existsSync('vendor/llama.cpp') ? '<(module_root_dir)/vendor/llama.cpp/build-apple/libllama.dylib' : '<(module_root_dir)/llama.cpp/build-apple/libllama.dylib'\")",
              "-framework Accelerate",
              "-framework Foundation",
              "-framework Metal"
            ]
          }
        ],
        [
          "OS=='linux'",
          {
            "cflags_cc": [
              "-std=c++20",
              "-fexceptions",
              "-frtti"
            ],
            "libraries": [
              "<!@(node -p \"require('fs').existsSync('vendor/llama.cpp') ? '<(module_root_dir)/vendor/llama.cpp/build-linux/libllama.so' : '<(module_root_dir)/llama.cpp/build-linux/libllama.so'\")"
            ],
            "ldflags": [
              "-Wl,-rpath,<!@(node -p \"require('fs').existsSync('vendor/llama.cpp') ? '<(module_root_dir)/vendor/llama.cpp/build-linux' : '<(module_root_dir)/llama.cpp/build-linux'\")"
            ]
          }
        ]
      ]
    }
  ]
}
