#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dex3_hands.hpp"
#include "hand_types.hpp"

struct HandStateSnapshot {
    bool has_state = false;
    std::vector<double> q;
    std::vector<double> dq;
};

class HandController {
public:
    virtual ~HandController() = default;

    virtual HandBackend backend() const = 0;
    virtual int dof() const = 0;
    virtual void initialize(const std::string& network_interface) = 0;
    virtual void setCommand(bool is_left, const std::vector<double>& values) = 0;
    virtual void writeOnce() = 0;
    virtual void open(bool is_left) = 0;
    virtual void close(bool is_left) = 0;
    virtual void stop(bool is_left) = 0;
    virtual void setMaxCloseRatio(double ratio) {
        (void)ratio;
    }
    virtual double getMaxCloseRatio() const {
        return 1.0;
    }
    virtual HandStateSnapshot getState(bool is_left) const = 0;
};

class NullHandController final : public HandController {
public:
    HandBackend backend() const override {
        return HandBackend::NONE;
    }

    int dof() const override {
        return 0;
    }

    void initialize(const std::string& network_interface) override {
        (void)network_interface;
    }

    void setCommand(bool is_left, const std::vector<double>& values) override {
        (void)is_left;
        (void)values;
    }

    void writeOnce() override {}

    void open(bool is_left) override {
        (void)is_left;
    }

    void close(bool is_left) override {
        (void)is_left;
    }

    void stop(bool is_left) override {
        (void)is_left;
    }

    HandStateSnapshot getState(bool is_left) const override {
        (void)is_left;
        return {};
    }
};

class Dex3HandController final : public HandController {
public:
    HandBackend backend() const override {
        return HandBackend::DEX3;
    }

    int dof() const override {
        return DEX3_MOTOR_MAX;
    }

    void initialize(const std::string& network_interface) override {
        hands_.initialize(network_interface);
    }

    void setCommand(bool is_left, const std::vector<double>& values) override {
        std::string error;
        if (!ValidateHandCommand(backend(), values, is_left ? "left hand command" : "right hand command", &error)) {
            throw std::runtime_error(error);
        }

        std::array<double, DEX3_MOTOR_MAX> command = {};
        std::copy(values.begin(), values.end(), command.begin());
        hands_.setAllJointsCommand(is_left, command);
    }

    void writeOnce() override {
        hands_.writeOnce();
    }

    void open(bool is_left) override {
        hands_.open(is_left);
    }

    void close(bool is_left) override {
        hands_.close(is_left);
    }

    void stop(bool is_left) override {
        hands_.stop(is_left);
    }

    void setMaxCloseRatio(double ratio) override {
        hands_.SetMaxCloseRatio(ratio);
    }

    double getMaxCloseRatio() const override {
        return hands_.GetMaxCloseRatio();
    }

    HandStateSnapshot getState(bool is_left) const override {
        HandStateSnapshot snapshot;
        const auto state_ptr = hands_.getState(is_left);
        snapshot.q.assign(DEX3_MOTOR_MAX, 0.0);
        snapshot.dq.assign(DEX3_MOTOR_MAX, 0.0);
        if (!state_ptr) {
            return snapshot;
        }

        snapshot.has_state = true;
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i) {
            snapshot.q[static_cast<size_t>(i)] = state_ptr->motor_state()[i].q();
            snapshot.dq[static_cast<size_t>(i)] = state_ptr->motor_state()[i].dq();
        }
        return snapshot;
    }

private:
    Dex3Hands hands_;
};

class InspireHandController final : public HandController {
public:
    explicit InspireHandController(bool dry_run) : dry_run_(dry_run) {}

    HandBackend backend() const override {
        return HandBackend::INSPIRE;
    }

    int dof() const override {
        return 6;
    }

    void initialize(const std::string& network_interface) override {
        (void)network_interface;
        if (!dry_run_) {
            throw std::runtime_error(
                "Inspire hand backend selected, but the hardware transport is not implemented. "
                "Re-run with --hand-dry-run to exercise the software path only.");
        }
        std::cout
            << "[InspireHandController] Running in explicit dry-run mode. "
            << "TODO: wire this backend to the Inspire hardware transport."
            << std::endl;
    }

    void setCommand(bool is_left, const std::vector<double>& values) override {
        std::string error;
        if (!ValidateHandCommand(backend(), values, is_left ? "left hand action" : "right hand action", &error)) {
            throw std::runtime_error(error);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_left) {
            left_command_ = values;
        } else {
            right_command_ = values;
        }
    }

    void writeOnce() override {
        if (!dry_run_) {
            return;
        }
    }

    void open(bool is_left) override {
        setCommand(is_left, std::vector<double>(static_cast<size_t>(dof()), 0.0));
    }

    void close(bool is_left) override {
        setCommand(is_left, std::vector<double>(static_cast<size_t>(dof()), 1.0));
    }

    void stop(bool is_left) override {
        open(is_left);
    }

    HandStateSnapshot getState(bool is_left) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        HandStateSnapshot snapshot;
        snapshot.has_state = dry_run_;
        snapshot.q = is_left ? left_command_ : right_command_;
        snapshot.dq.assign(snapshot.q.size(), 0.0);
        return snapshot;
    }

private:
    bool dry_run_ = false;
    mutable std::mutex mutex_;
    std::vector<double> left_command_ = std::vector<double>(6, 0.0);
    std::vector<double> right_command_ = std::vector<double>(6, 0.0);
};

inline std::unique_ptr<HandController> CreateHandController(
    HandBackend backend,
    bool dry_run) {
    switch (backend) {
        case HandBackend::NONE:
            return std::make_unique<NullHandController>();
        case HandBackend::DEX3:
            return std::make_unique<Dex3HandController>();
        case HandBackend::INSPIRE:
            return std::make_unique<InspireHandController>(dry_run);
    }
    return std::make_unique<NullHandController>();
}
