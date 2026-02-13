/**
 * BranchStore - Batched multi-branch decode operations
 *
 * See index.d.ts for full API documentation.
 */
class BranchStore {
  constructor(ctx) {
    this._ctx = ctx;
  }

  // entries: [branch, token][] — binding is structural, not positional
  commit(entries) {
    const handles = [], tokens = [];
    for (const [branch, token] of entries) {
      if (branch.disposed) throw new Error('BranchStore.commit: branch is disposed');
      handles.push(branch.handle);
      tokens.push(token);
    }
    this._ctx._storeCommit(handles, tokens);
  }

  // entries: [branch, tokens[]][] — binding is structural, not positional
  prefill(entries) {
    const handles = [], tokenArrays = [];
    for (const [branch, tokens] of entries) {
      if (branch.disposed) throw new Error('BranchStore.prefill: branch is disposed');
      handles.push(branch.handle);
      tokenArrays.push(tokens);
    }
    this._ctx._storePrefill(handles, tokenArrays);
  }
}

module.exports = { BranchStore };
