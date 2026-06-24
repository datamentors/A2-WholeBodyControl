#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

enum class HandBackend {
    NONE,
    DEX3,
    INSPIRE,
};

struct HandCommand {
    HandBackend backend = HandBackend::NONE;
    std::vector<double> values;
};

inline std::string HandBackendToString(HandBackend backend) {
    switch (backend) {
        case HandBackend::NONE:
            return "none";
        case HandBackend::DEX3:
            return "dex3";
        case HandBackend::INSPIRE:
            return "inspire";
    }
    return "unknown";
}

inline std::optional<HandBackend> ParseHandBackend(const std::string& raw_value) {
    std::string value = raw_value;
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (value == "none") {
        return HandBackend::NONE;
    }
    if (value == "dex3") {
        return HandBackend::DEX3;
    }
    if (value == "inspire") {
        return HandBackend::INSPIRE;
    }
    return std::nullopt;
}

inline int HandBackendDof(HandBackend backend) {
    switch (backend) {
        case HandBackend::NONE:
            return 0;
        case HandBackend::DEX3:
            return 7;
        case HandBackend::INSPIRE:
            return 6;
    }
    return 0;
}

inline std::string FormatHandValues(const std::vector<double>& values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

inline bool ValidateHandCommand(
    HandBackend backend,
    const std::vector<double>& values,
    const std::string& label,
    std::string* error) {
    const int expected_dof = HandBackendDof(backend);
    if (backend == HandBackend::NONE) {
        if (!values.empty()) {
            if (error != nullptr) {
                *error = label + " must be empty when hand backend is 'none'";
            }
            return false;
        }
        return true;
    }

    if (static_cast<int>(values.size()) != expected_dof) {
        if (error != nullptr) {
            *error = label + " must have " + std::to_string(expected_dof) +
                     " values for hand backend '" + HandBackendToString(backend) +
                     "', got " + std::to_string(values.size());
        }
        return false;
    }
    return true;
}
