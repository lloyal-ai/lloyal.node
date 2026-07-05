/**
 * Backend-pack signature + integrity verification.
 *
 * The binaries channel (apps.lloyal.ai/v1/binaries/…) publishes each pack
 * as write-once archives plus a platform-signed manifest. The publish
 * worker signs the EXACT canonical-JSON bytes it stores, so verification
 * here is over the raw fetched manifest body — no canonical-JSON
 * re-encoding exists on the consumer side, and there is nothing to drift.
 *
 * Self-contained by design (decision 2026-07-06): the pinned public key is
 * a verbatim copy of `LLOYAL_PLATFORM_KEY_2026_Q2` shared by
 * `@lloyal-labs/rig` and `harness.dev`; consolidation into a shared
 * `@lloyal-labs/channel-verify` package is tracked separately (hdk #465)
 * and absorbs this file when it lands.
 */

import { createHash, createPublicKey, verify as cryptoVerify } from 'node:crypto';

/**
 * The current Lloyal platform Ed25519 public key (raw 32 bytes) —
 * `lloyal-platform-2026-q2`. Verbatim copy of `LLOYAL_PLATFORM_KEY_2026_Q2`
 * in `@lloyal-labs/rig/src/protocol.ts`.
 *
 * SHA-256 fingerprint: 9e0df3d25b8968a8b2ae9b86cb17a6922368c7cff9674a84b4a2527dd6457ec1
 * Base64: bUz2SCkISzbzD4/WftUw4Nou2bJixs6OYh/5lomQylI=
 */
const LLOYAL_PLATFORM_KEY_2026_Q2 = new Uint8Array([
  109, 76, 246, 72, 41, 8, 75, 54, 243, 15, 143, 214, 126, 213, 48, 224,
  218, 46, 217, 178, 98, 198, 206, 142, 98, 31, 249, 150, 137, 144, 202, 82,
]);

/** DER prefix that wraps a raw Ed25519 public key into SPKI (RFC 8410). */
const ED25519_SPKI_PREFIX = new Uint8Array([
  0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
]);

/**
 * Verify an Ed25519 signature (base64) over `bytes` against the pinned
 * platform key. Returns boolean; throwing is the caller's policy.
 */
export function verifyPlatformSignature(
  bytes: Uint8Array,
  signatureB64: string,
  rawPublicKey: Uint8Array = LLOYAL_PLATFORM_KEY_2026_Q2,
): boolean {
  const spki = Buffer.concat([ED25519_SPKI_PREFIX, rawPublicKey]);
  const key = createPublicKey({ key: spki, format: 'der', type: 'spki' });
  const signature = Buffer.from(signatureB64, 'base64');
  if (signature.length !== 64) return false;
  return cryptoVerify(null, bytes, key, signature);
}

/** Lowercase-hex sha256 of a byte buffer. */
export function sha256Hex(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}
