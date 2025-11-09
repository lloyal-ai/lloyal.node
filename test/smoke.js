/**
 * Smoke test - verify the addon loads and exports are present
 */

const path = require('path');

console.log('Loading liblloyal-node addon...');

try {
  // Load the built addon directly
  const addon = require('../build/Release/lloyal_node.node');

  console.log('✓ Addon loaded successfully');
  console.log('Exports:', Object.keys(addon));

  // Check for expected exports
  const hasCreateContext = typeof addon.createContext === 'function';
  const hasSessionContext = typeof addon.SessionContext === 'function';

  console.log('✓ createContext:', hasCreateContext ? 'present' : 'MISSING');
  console.log('✓ SessionContext:', hasSessionContext ? 'present' : 'MISSING');

  if (hasCreateContext && hasSessionContext) {
    console.log('\n✅ All exports present - scaffold is live!');
    process.exit(0);
  } else {
    console.log('\n❌ Missing expected exports');
    process.exit(1);
  }
} catch (err) {
  console.error('❌ Failed to load addon:', err.message);
  console.error(err.stack);
  process.exit(1);
}
