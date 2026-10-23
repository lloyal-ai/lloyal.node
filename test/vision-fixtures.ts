import { resolve } from 'node:path';

const models = resolve(__dirname, '../models');

/** Qwen3.5-4B with its vision projector: the model image and video tests read with, unless the environment names another. */
export const visionModel = {
  modelPath: resolve(process.env.LLAMA_VL_MODEL ?? resolve(models, 'Qwen3.5-4B-Q4_K_M.gguf')),
  mmprojPath: resolve(process.env.LLAMA_VL_MMPROJ ?? resolve(models, 'mmproj-Qwen3.5-4B-F16.gguf')),
};
