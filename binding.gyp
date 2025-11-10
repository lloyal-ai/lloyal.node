{
  "targets": [
    {
      "target_name": "liblloyal_node",
      "sources": [
        "src/binding.cpp",
        "src/BackendManager.cpp",
        "src/SessionContext.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "liblloyal/include",
        "liblloyal/include/lloyal",
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
              "LD_RUNPATH_SEARCH_PATHS": [
                "@loader_path/../../llama.cpp/build-apple/llama.xcframework/macos-arm64_x86_64",
                "@executable_path/../lib"
              ]
            },
            "libraries": [
              "-F<(module_root_dir)/llama.cpp/build-apple/llama.xcframework/macos-arm64_x86_64",
              "-framework llama",
              "-framework Accelerate",
              "-framework Foundation",
              "-framework Metal",
              "-framework MetalKit"
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
              "<(module_root_dir)/llama.cpp/build-linux/libllama.so"
            ],
            "ldflags": [
              "-Wl,-rpath,<(module_root_dir)/llama.cpp/build-linux"
            ]
          }
        ]
      ]
    }
  ]
}
