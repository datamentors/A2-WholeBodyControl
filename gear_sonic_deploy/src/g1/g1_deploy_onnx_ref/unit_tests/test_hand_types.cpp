#include <gtest/gtest.h>

#include "../include/hand_types.hpp"

TEST(HandTypes, ParsesBackendNames) {
    ASSERT_TRUE(ParseHandBackend("dex3").has_value());
    EXPECT_EQ(*ParseHandBackend("dex3"), HandBackend::DEX3);

    ASSERT_TRUE(ParseHandBackend("Inspire").has_value());
    EXPECT_EQ(*ParseHandBackend("Inspire"), HandBackend::INSPIRE);

    ASSERT_TRUE(ParseHandBackend("none").has_value());
    EXPECT_EQ(*ParseHandBackend("none"), HandBackend::NONE);

    EXPECT_FALSE(ParseHandBackend("mystery").has_value());
}

TEST(HandTypes, ReportsExpectedDofPerBackend) {
    EXPECT_EQ(HandBackendDof(HandBackend::DEX3), 7);
    EXPECT_EQ(HandBackendDof(HandBackend::INSPIRE), 6);
    EXPECT_EQ(HandBackendDof(HandBackend::NONE), 0);
}

TEST(HandTypes, ValidatesBackendSpecificShapes) {
    std::string error;
    EXPECT_TRUE(
        ValidateHandCommand(HandBackend::DEX3, std::vector<double>(7, 0.0), "dex3", &error));
    EXPECT_TRUE(
        ValidateHandCommand(HandBackend::INSPIRE, std::vector<double>(6, 0.0), "inspire", &error));
    EXPECT_TRUE(
        ValidateHandCommand(HandBackend::NONE, {}, "none", &error));

    EXPECT_FALSE(
        ValidateHandCommand(HandBackend::DEX3, std::vector<double>(6, 0.0), "dex3", &error));
    EXPECT_NE(error.find("7 values"), std::string::npos);

    EXPECT_FALSE(
        ValidateHandCommand(HandBackend::INSPIRE, std::vector<double>(7, 0.0), "inspire", &error));
    EXPECT_NE(error.find("6 values"), std::string::npos);
}
