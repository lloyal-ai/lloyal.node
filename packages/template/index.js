// Platform-specific binary package template
// This file exports the path to the native binary in bin/

const path = require('path');

module.exports = path.join(__dirname, 'bin', 'lloyal.node');
