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

function requireLegalFile(source) {
  if (!fs.existsSync(source)) {
    throw new Error(`Missing legal document: ${source}. Ensure the source checkout and toolkit are complete before packaging.`);
  }
}

function validateNativeLegalFiles(root) {
  for (const file of NATIVE_LEGAL_FILES) requireLegalFile(path.join(root, file));
  // Check our sources, including the pinned kernel, before creating any payload.
  // Third-party licenses are preserved verbatim and are not subject to this gate.
  for (const file of ['LICENSE', 'liblloyal/LICENSE']) {
    const license = fs.readFileSync(path.join(root, file), 'utf8');
    if (!/^# Functional Source License, Version 1\.1, MIT Future License\r?\n/.test(license)
        || !/\r?\nFSL-1\.1-MIT\r?\n/.test(license)) {
      throw new Error(`Expected FSL-1.1-MIT in ${file}. Check the root license and pinned liblloyal revision before packaging.`);
    }
  }
}

function copyFile(source, destination) {
  requireLegalFile(source);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

function copyNativeLegalFiles(root, destination) {
  validateNativeLegalFiles(root);
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

module.exports = { ROOT_LEGAL_FILES, NATIVE_LEGAL_FILES, validateNativeLegalFiles, copyNativeLegalFiles, copyCudaLegalFiles };
