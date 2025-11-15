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
                "-Wl,-rpath,@loader_path"
              ]
            },
            "libraries": [
              "<(module_root_dir)/build/Release/libllama.dylib",
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
              "<(module_root_dir)/build/Release/libllama.so"
            ],
            "ldflags": [
              "-Wl,-rpath,$$ORIGIN"
            ]
          }
        ],
        [
          "OS=='win'",
          {
            "msvs_settings": {
              "VCCLCompilerTool": {
                "ExceptionHandling": 1,
                "RuntimeTypeInfo": "true",
                "AdditionalOptions": [
                  "/std:c++20",
                  "/GR"
                ]
              }
            },
            "libraries": [
              "<(module_root_dir)/build/Release/llama.lib"
            ]
          }
        ]
      ]
    }
  ]
}
