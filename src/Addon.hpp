#pragma once

#include <napi.h>

/**
 * What the addon keeps per environment: the constructors its factories call.
 * `createContext` makes a SessionContext; `formatChat` makes a ChatParser.
 */
struct AddonData {
  Napi::FunctionReference contextCtor;
  Napi::FunctionReference parserCtor;
};
