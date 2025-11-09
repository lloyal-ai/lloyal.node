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
        "include"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": ["-fno-exceptions", "-fno-rtti"],
      "cflags_cc!": ["-fno-exceptions", "-fno-rtti"],
      "defines": [],
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
              ]
            },
            "libraries": [
              "<(module_root_dir)/llama.cpp/build-macos/src/libllama.a",
              "<(module_root_dir)/llama.cpp/build-macos/ggml/src/libggml-base.a",
              "<(module_root_dir)/llama.cpp/build-macos/ggml/src/libggml-cpu.a",
              "<(module_root_dir)/llama.cpp/build-macos/ggml/src/ggml-metal/libggml-metal.a",
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
              "<(module_root_dir)/llama.cpp/build/src/libllama.a",
              "<(module_root_dir)/llama.cpp/build/ggml/src/libggml-base.a",
              "<(module_root_dir)/llama.cpp/build/ggml/src/libggml-cpu.a"
            ]
          }
        ]
      ]
    }
  ]
}
