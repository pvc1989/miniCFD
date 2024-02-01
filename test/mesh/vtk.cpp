// Copyright 2024 PEI Weicheng
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/vtk.hpp"

class TestMeshVtk : public ::testing::Test {
 protected:
};
TEST_F(TestMeshVtk, EncodeBase64) {
  using mini::mesh::vtk::EncodeBase64;
  std::string origin, encoded;
  origin = "A";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QQ==");
  origin = "AB";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUI=");
  origin = "ABC";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJD");
  origin = "ABCD";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDRA==");
  origin = "ABCDE";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDREU=");
  origin = "ABCDEF";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDREVG");
  origin = "1";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MQ==");
  origin = "12";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTI=");
  origin = "123";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIz");
  origin = "1234";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNA==");
  origin = "12345";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNDU=");
  origin = "123456";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNDU2");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
