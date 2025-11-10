#pragma once

#include <string>
#include <cstdio>
#include <cerrno>

namespace margelo::nitro::nitrollama {

/**
 * File system operations and validation service
 * Centralizes all file-related operations with consistent error handling
 */
namespace FileSystem {

    /**
     * Check if a file exists and is accessible
     * @param path File path to check
     * @return true if file exists and can be read
     */
    inline bool exists(const std::string& path) {
        FILE* file = fopen(path.c_str(), "rb");
        if (file) {
            fclose(file);
            return true;
        }
        return false;
    }

    /**
     * Get file size in bytes
     * @param path File path to check
     * @return File size in bytes, or 0 if file doesn't exist
     */
    inline size_t getSize(const std::string& path) {
        FILE* file = fopen(path.c_str(), "rb");
        if (!file) {
            return 0;
        }

        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fclose(file);

        return size > 0 ? static_cast<size_t>(size) : 0;
    }

    /**
     * Convert file:// URI to filesystem path
     * @param path Original path (may be file:// URI or regular path)
     * @return Filesystem path without file:// prefix
     */
    inline std::string normalizePath(const std::string& path) {
        if (path.find("file://") == 0) {
            return path.substr(7); // Remove "file://" prefix
        }
        return path;
    }

    /**
     * Validate a model file with detailed status
     * @param path File path to validate
     * @return Detailed validation result string
     */
    inline std::string validate(const std::string& path) {
        std::string result = "Validating: " + path + "\n";

        // Normalize path
        std::string fsPath = normalizePath(path);
        if (fsPath != path) {
            result += "Converted file:// URI to: " + fsPath + "\n";
        }

        // Check existence and access
        if (!exists(fsPath)) {
            result += "ERROR: Cannot access file (errno: " + std::to_string(errno) + ")\n";
            return result;
        }

        // Get file size
        size_t fileSize = getSize(fsPath);
        result += "File accessible, size: " + std::to_string(fileSize) + " bytes\n";

        // Basic validation
        if (fileSize < 1024) {
            result += "WARNING: File is very small (< 1KB)\n";
        } else {
            result += "File size looks reasonable\n";
        }

        return result;
    }

    /**
     * Throw appropriate exception for file access errors
     * @param path File path that failed
     * @param operation Description of the attempted operation
     */
    inline void throwAccessError(const std::string& path, const std::string& operation) {
        throw std::runtime_error(
            operation + " failed for: " + path +
            " (errno: " + std::to_string(errno) + ")"
        );
    }
}

} // namespace margelo::nitro::nitrollama
