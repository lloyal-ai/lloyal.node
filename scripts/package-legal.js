/** Legal documents travel with each binary payload, verbatim from its sources. */
const fs = require('fs');
const path = require('path');

const ROOT_LEGAL_FILES = ['LICENSE', 'NOTICE', 'GRANT.md', 'LICENSE-FAQ.md'];
const NATIVE_LEGAL_FILES = [
  ...ROOT_LEGAL_FILES,
  'liblloyal/LICENSE',
  'liblloyal/NOTICE',
  'llama.cpp/LICENSE',
];

function copyFile(source, destination) {
  if (!fs.existsSync(source)) {
    throw new Error(`Missing legal document: ${source}. Ensure the source checkout and toolkit are complete before packaging.`);
  }
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

function copyNativeLegalFiles(root, destination) {
  for (const file of NATIVE_LEGAL_FILES) {
    copyFile(path.join(root, file), path.join(destination, file));
  }
  // Preserve an upstream NOTICE if the pinned llama.cpp revision supplies one.
  const notice = 'llama.cpp/NOTICE';
  if (fs.existsSync(path.join(root, notice))) {
    copyFile(path.join(root, notice), path.join(destination, notice));
  }
}

function copyCudaLegalFiles(cudaPath, destination) {
  // The CUDA toolkit installed by provision-cuda carries NVIDIA's own EULA.
  // Keep it separate: Lloyal's FSL/Grant does not license the companion libs.
  copyFile(path.join(cudaPath, 'EULA.txt'), path.join(destination, 'third-party/cuda/EULA.txt'));
}

module.exports = { ROOT_LEGAL_FILES, NATIVE_LEGAL_FILES, copyNativeLegalFiles, copyCudaLegalFiles };
